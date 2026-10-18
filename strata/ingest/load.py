"""``strata ingest``: read a source through its mapping, check it, write the raw layer.

A source is described by a mapping (``strata.contracts.mapping``, ``mappings/<source>.yaml``). The
ingest writes:

- ``raw.clients``, ``raw.events`` and ``raw.client_aggregates``: the canonical tables, the only
  ones dbt staging reads;
- ``landing.<extract>``: for files read through a typed contract of the reference extract, the file
  exactly as parsed (original column names, ``_source_row``), which the source-specific
  data-quality checks of ``strata dq`` read;
- ``raw.manifest``: one row per file, with its role.

Zone rules (ADR 0008):

- a **public** run accepts only a folder whose synthetic marker matches its files byte for byte, and
  writes to the public warehouse;
- a **private** run reads ``STRATA_DATA_DIR`` (or an explicit folder outside the repository) and
  writes only under ``STRATA_PRIVATE_DIR``.

After loading, reconciliation compares the warehouse with the parsed files (row counts, amount sums,
distinct clients), for the landing tables and for the canonical ones, so that a silent loss between
pandas and DuckDB, or in the mapping, cannot go unnoticed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from strata.config import PrivacyBoundaryError, Settings, Workspace, Zone, is_within
from strata.contracts.extracts import AMOUNT_COLUMNS, ColumnType
from strata.contracts.mapping import (
    AGGREGATE_TYPES,
    CLIENT_TYPES,
    EVENT_COLUMNS,
    EVENT_TYPES,
    REFERENCE_MAPPING,
    SYNTHETIC_MAPPING,
    EventFile,
    LoadedMapping,
    SourceMapping,
    TableFile,
    canonical_events,
    canonical_table,
    file_settings,
    load_mapping,
    source_columns,
)
from strata.contracts.specs import EXTRACTS, Extract, spec_for
from strata.ingest.manifest import FileManifest, Manifest
from strata.ingest.readers import read_delimited, read_extract
from strata.log import get_logger
from strata.synthetic.writer import marker_matches, read_marker
from strata.warehouse import connect, read_table, replace_table, row_count

log = get_logger(__name__)

RAW = "raw"
LANDING = "landing"
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


@dataclass(frozen=True, slots=True)
class LoadedFile:
    """One file of the source: its role, its parsed frame, and its manifest entry."""

    role: str
    label: str
    frame: pd.DataFrame
    manifest: FileManifest
    contract: Extract | None
    mapping: TableFile | EventFile


def default_mapping(zone: Zone) -> Path:
    """The mapping of the reference instance in the private zone, of the generator in public."""
    return REFERENCE_MAPPING if zone is Zone.PRIVATE else SYNTHETIC_MAPPING


def resolve_source(
    settings: Settings, zone: Zone, data_dir: Path | None, files: list[str] | None = None
) -> Source:
    """Decide what ``data_dir`` is and whether this zone may read it.

    ``files`` are the source's files (from its mapping); by default those of the reference extract.
    """
    files = files if files is not None else [spec.filename for spec in EXTRACTS]
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
    if not marker_matches(marker, path, files):
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


def _date_range(frame: pd.DataFrame, column: str | None) -> tuple[date | None, date | None]:
    if column is None or not len(frame):
        return None, None
    dates = pd.to_datetime(frame[column])
    return dates.min().date(), dates.max().date()


def _read_file(
    folder: Path,
    mapping: SourceMapping,
    file: TableFile | EventFile,
    *,
    role: str,
    label: str,
    types: dict[str, ColumnType],
) -> LoadedFile:
    """Read one file, through its contract when it has one, generically otherwise."""
    path = folder / file.file
    date_column = file.columns.get("ts") if isinstance(file, EventFile) else None
    if file.contract is not None:
        spec = spec_for(file.contract)
        result = read_extract(path, spec)
        date_min, date_max = _date_range(result.frame, spec.date_column)
        entry = FileManifest(
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
        return LoadedFile(role, spec.name.value, result.frame, entry, file.contract, file)
    read = read_delimited(
        path, label=label, columns=source_columns(file, types), **file_settings(mapping, file)
    )
    date_min, date_max = _date_range(read.frame, date_column)
    entry = FileManifest(
        extract=label,
        filename=file.file,
        sha256=read.sha256,
        size_bytes=read.size_bytes,
        rows=len(read.frame),
        columns=len(read.header),
        line_terminator=read.line_terminator,
        date_min=date_min,
        date_max=date_max,
    )
    return LoadedFile(role, label, read.frame, entry, None, file)


def read_source(folder: Path, mapping: SourceMapping) -> list[LoadedFile]:
    """Every file of the source: clients, then events, then aggregates (the manifest's order)."""
    loaded = [
        _read_file(
            folder, mapping, mapping.clients, role="clients", label="clients", types=CLIENT_TYPES
        )
    ]
    for index, event in enumerate(mapping.events, start=1):
        label = "events" if len(mapping.events) == 1 else f"events_{index}"
        loaded.append(
            _read_file(folder, mapping, event, role="events", label=label, types=EVENT_TYPES)
        )
    if mapping.client_aggregates is not None:
        loaded.append(
            _read_file(
                folder,
                mapping,
                mapping.client_aggregates,
                role="client_aggregates",
                label="client_aggregates",
                types=AGGREGATE_TYPES,
            )
        )
    return loaded


def canonical_frames(files: list[LoadedFile]) -> dict[str, pd.DataFrame]:
    """The canonical tables of a source (an absent aggregate file gives an empty table)."""
    tables: dict[str, pd.DataFrame] = {}
    events = []
    for item in files:
        if isinstance(item.mapping, EventFile):
            events.append(canonical_events(item.frame, item.mapping, label=item.label))
        else:
            types = CLIENT_TYPES if item.role == "clients" else AGGREGATE_TYPES
            tables[item.role] = canonical_table(item.frame, item.mapping, types)
    tables["events"] = pd.concat(events, ignore_index=True)[list(EVENT_COLUMNS)]
    if "client_aggregates" not in tables:
        empty = TableFile(file="", columns={"client_id": "client_id"})
        tables["client_aggregates"] = canonical_table(
            pd.DataFrame({"client_id": pd.Series([], dtype="object")}), empty, AGGREGATE_TYPES
        )
    return tables


def _with_source_row(frame: pd.DataFrame) -> pd.DataFrame:
    """Position of each row in its file (1-based), kept for traceability and stable ordering."""
    return frame.assign(_source_row=range(1, len(frame) + 1))


def reconcile(
    con: duckdb.DuckDBPyConnection, files: list[LoadedFile], tables: dict[str, pd.DataFrame]
) -> list[Reconciliation]:
    """Compare the landing and canonical tables with the frames they were loaded from."""
    checks: list[Reconciliation] = []
    for item in files:
        if item.contract is None:
            continue
        spec = spec_for(item.contract)
        stored = read_table(con, LANDING, item.contract.value)
        frame = item.frame
        name = item.contract.value
        checks.append(Reconciliation(name, "rows", float(len(frame)), float(len(stored))))
        checks.append(
            Reconciliation(
                name,
                "distinct_ids",
                float(frame[spec.id_column].nunique()),
                float(stored[spec.id_column].nunique()),
            )
        )
        amount = AMOUNT_COLUMNS.get(item.contract)
        if amount is not None:
            checks.append(
                Reconciliation(
                    name, f"sum_{amount}", float(frame[amount].sum()), float(stored[amount].sum())
                )
            )
    events = read_table(con, RAW, "events")
    for item in files:
        if item.role != "events":
            continue
        rows = events[events["source_file"] == item.mapping.file]
        checks.append(
            Reconciliation(f"events:{item.label}", "rows", float(len(item.frame)), float(len(rows)))
        )
        if "amount" in item.mapping.columns:
            source = item.frame[item.mapping.columns["amount"]]
            checks.append(
                Reconciliation(
                    f"events:{item.label}",
                    "sum_amount",
                    float(source.sum()),
                    float(rows["amount"].sum()),
                )
            )
    for role in ("clients", "client_aggregates"):
        expected = float(len(tables[role]))
        checks.append(
            Reconciliation(role, "canonical_rows", expected, float(row_count(con, RAW, role)))
        )
    return checks


def ingest(
    settings: Settings,
    zone: Zone,
    data_dir: Path | None = None,
    mapping: Path | str | None = None,
) -> IngestReport:
    """Load the source in ``data_dir``, described by ``mapping``, into the warehouse of ``zone``."""
    loaded: LoadedMapping = load_mapping(mapping or default_mapping(zone), settings.repo_root)
    source = resolve_source(settings, zone, data_dir, loaded.mapping.files)
    workspace = settings.workspace(zone)
    if workspace.is_private:
        settings.assert_private_output(workspace.warehouse)
    files = read_source(source.path, loaded.mapping)
    tables = canonical_frames(files)
    manifest = Manifest(
        source_kind="real" if source.kind == "real" else "synthetic",
        generator=source.generator,
        files=[item.manifest for item in files],
        loaded_at=datetime.now(UTC),
        mapping=loaded.mapping.source,
        mapping_sha256=loaded.sha256,
    )
    roles = {item.manifest.extract: item.role for item in files}
    con = connect(workspace.warehouse)
    try:
        con.execute(f"drop schema if exists {LANDING} cascade")
        for item in files:
            if item.contract is not None:
                replace_table(con, LANDING, item.contract.value, _with_source_row(item.frame))
        for role, frame in tables.items():
            replace_table(con, RAW, role, frame)
        manifest_rows = pd.DataFrame(
            [
                {
                    **item.model_dump(),
                    "role": roles[item.extract],
                    "source_kind": manifest.source_kind,
                    "fingerprint": manifest.fingerprint,
                    "mapping": manifest.mapping,
                }
                for item in manifest.files
            ]
        )
        replace_table(con, RAW, "manifest", manifest_rows)
        reconciliations = reconcile(con, files, tables)
    finally:
        con.close()
    workspace.root.mkdir(parents=True, exist_ok=True)
    (workspace.root / MANIFEST_FILE).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline=""
    )
    log.info(
        "ingested",
        zone=zone.value,
        files=len(files),
        source_kind=manifest.source_kind,
        mapping=manifest.mapping,
    )
    return IngestReport(workspace=workspace, manifest=manifest, reconciliations=reconciliations)


def load_manifest(workspace: Workspace) -> Manifest:
    """The manifest of the last ingest into ``workspace``."""
    path = workspace.root / MANIFEST_FILE
    if not path.is_file():
        msg = "No manifest found: run `strata ingest` first."
        raise FileNotFoundError(msg)
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
