"""``strata ingest``: read the five files, check them, and write the warehouse's raw layer.

Zone rules (ADR 0008):

- a **public** run accepts only a folder whose synthetic marker matches its files byte for byte, and
  writes to the public warehouse;
- a **private** run reads ``STRATA_DATA_DIR`` (or an explicit folder outside the repository) and
  writes only under ``STRATA_PRIVATE_DIR``.

After loading, reconciliation compares the warehouse with the parsed files (row counts, amount sums,
distinct clients) so that a silent loss between pandas and DuckDB cannot go unnoticed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from strata.config import PrivacyBoundaryError, Settings, Workspace, Zone, is_within
from strata.contracts.extracts import AMOUNT_COLUMNS
from strata.contracts.specs import EXTRACTS, Extract
from strata.ingest.manifest import FileManifest, Manifest
from strata.ingest.readers import ReadResult, read_extract
from strata.log import get_logger
from strata.synthetic.writer import marker_matches, read_marker
from strata.warehouse import connect, read_table, replace_table

log = get_logger(__name__)

RAW = "raw"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True, slots=True)
class Source:
    """A data folder and what it is."""

    path: Path
    kind: str
    generator: dict[str, str | int] | None


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """One comparison between the parsed files and the warehouse."""

    extract: str
    measure: str
    source: float
    warehouse: float

    @property
    def ok(self) -> bool:
        return math.isclose(self.source, self.warehouse, rel_tol=1e-9, abs_tol=1e-6)


@dataclass(frozen=True, slots=True)
class IngestReport:
    workspace: Workspace
    manifest: Manifest
    reconciliations: list[Reconciliation]

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.reconciliations)


def resolve_source(settings: Settings, zone: Zone, data_dir: Path | None) -> Source:
    """Decide what ``data_dir`` is and whether this zone may read it."""
    if zone is Zone.PRIVATE:
        path = (data_dir or settings.require_data_dir()).expanduser().resolve()
        marker = read_marker(path)
        if marker is None and is_within(path, settings.repo_root):
            msg = "Refusing to read unmarked data from inside the repository."
            raise PrivacyBoundaryError(msg)
        generator: dict[str, str | int] | None = (
            None if marker is None else {"seed": marker.seed, "n_clients": marker.n_clients}
        )
        return Source(
            path=path, kind="real" if marker is None else "synthetic", generator=generator
        )
    if data_dir is None:
        msg = "A public run needs --data-dir pointing at output of `strata synth`."
        raise PrivacyBoundaryError(msg)
    path = data_dir.expanduser().resolve()
    marker = read_marker(path)
    if marker is None:
        msg = (
            "This folder has no synthetic marker. Real data is ingested only with --private, "
            "and its outputs go to STRATA_PRIVATE_DIR."
        )
        raise PrivacyBoundaryError(msg)
    if not marker_matches(marker, path, (spec.filename for spec in EXTRACTS)):
        msg = (
            "The files no longer match their synthetic marker; regenerate them with `strata synth`."
        )
        raise PrivacyBoundaryError(msg)
    found: dict[str, str | int] = {
        "seed": marker.seed,
        "n_clients": marker.n_clients,
        "digest": marker.config_digest,
    }
    return Source(path=path, kind="synthetic", generator=found)


def _file_manifest(result: ReadResult) -> FileManifest:
    spec = result.spec
    date_min = date_max = None
    if spec.date_column is not None and len(result.frame):
        dates = pd.to_datetime(result.frame[spec.date_column])
        date_min, date_max = dates.min().date(), dates.max().date()
    return FileManifest(
        extract=spec.name.value,
        filename=spec.filename,
        sha256=result.sha256,
        size_bytes=result.size_bytes,
        rows=len(result.frame),
        columns=len(spec.columns),
        line_terminator=result.line_terminator,
        date_min=date_min,
        date_max=date_max,
    )


def _with_source_row(frame: pd.DataFrame) -> pd.DataFrame:
    """Position of each row in its file (1-based), kept for traceability and stable ordering."""
    return frame.assign(_source_row=range(1, len(frame) + 1))


def reconcile(
    con: duckdb.DuckDBPyConnection, results: dict[Extract, ReadResult]
) -> list[Reconciliation]:
    """Compare every raw table with the frame it was loaded from."""
    checks: list[Reconciliation] = []
    for name, result in results.items():
        stored = read_table(con, RAW, name.value)
        frame = result.frame
        checks.append(Reconciliation(name.value, "rows", float(len(frame)), float(len(stored))))
        id_column = result.spec.id_column
        checks.append(
            Reconciliation(
                name.value,
                "distinct_ids",
                float(frame[id_column].nunique()),
                float(stored[id_column].nunique()),
            )
        )
        amount = AMOUNT_COLUMNS.get(name)
        if amount is not None:
            checks.append(
                Reconciliation(
                    name.value,
                    f"sum_{amount}",
                    float(frame[amount].sum()),
                    float(stored[amount].sum()),
                )
            )
    return checks


def ingest(settings: Settings, zone: Zone, data_dir: Path | None = None) -> IngestReport:
    """Load the five files of ``data_dir`` into the warehouse of ``zone``."""
    source = resolve_source(settings, zone, data_dir)
    workspace = settings.workspace(zone)
    if workspace.is_private:
        settings.assert_private_output(workspace.warehouse)
    results = {spec.name: read_extract(source.path / spec.filename, spec) for spec in EXTRACTS}
    manifest = Manifest(
        source_kind="real" if source.kind == "real" else "synthetic",
        generator=source.generator,
        files=[_file_manifest(result) for result in results.values()],
        loaded_at=datetime.now(UTC),
    )
    con = connect(workspace.warehouse)
    try:
        for name, result in results.items():
            replace_table(con, RAW, name.value, _with_source_row(result.frame))
        manifest_rows = pd.DataFrame(
            [
                {
                    **item.model_dump(),
                    "source_kind": manifest.source_kind,
                    "fingerprint": manifest.fingerprint,
                }
                for item in manifest.files
            ]
        )
        replace_table(con, RAW, "manifest", manifest_rows)
        reconciliations = reconcile(con, results)
    finally:
        con.close()
    workspace.root.mkdir(parents=True, exist_ok=True)
    (workspace.root / MANIFEST_FILE).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline=""
    )
    log.info("ingested", zone=zone.value, files=len(results), source_kind=manifest.source_kind)
    return IngestReport(workspace=workspace, manifest=manifest, reconciliations=reconciliations)


def load_manifest(workspace: Workspace) -> Manifest:
    """The manifest of the last ingest into ``workspace``."""
    path = workspace.root / MANIFEST_FILE
    if not path.is_file():
        msg = "No manifest found: run `strata ingest` first."
        raise FileNotFoundError(msg)
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
