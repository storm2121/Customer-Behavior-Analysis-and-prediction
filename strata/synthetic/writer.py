"""Write synthetic frames as files that are byte-for-byte shaped like the extract.

Format facts reproduced (ARCHITECTURE.md section 5.1 and the contracts): Latin-1 encoding, CRLF line
endings with a final line break, no quoting (no field contains a separator), dates written
``YYYY-MM-DDTHH:MM`` at midnight, whole amounts written without a decimal point, and the aggregate
file's empty row before every data row.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from strata import __version__
from strata.config import SYNTHETIC_MARKER, PrivacyBoundaryError, is_within
from strata.contracts.extracts import COLUMN_TYPES, ColumnType
from strata.contracts.specs import (
    DATE_FORMAT,
    ENCODING,
    EXTRACTS,
    LINE_TERMINATOR,
    Extract,
    ExtractSpec,
)
from strata.synthetic.parametric import GENERATOR_VERSION, SyntheticExtract, config_digest

TRUTH_FILE = "synthetic_truth.csv"
"""Latent traits behind the synthetic clients, for evaluating models against known structure."""

_FORBIDDEN = (",", '"', "\r", "\n")


def _trim(text: str) -> str:
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_float(value: float, decimals: int) -> str:
    """``1600.0`` → ``1600``, ``20.50`` → ``20.5``, NaN → empty."""
    if math.isnan(value):
        return ""
    text = _trim(f"{value:.{decimals}f}")
    return "0" if text in {"-0", ""} else text


def _format_column(series: pd.Series, kind: ColumnType, decimals: int) -> list[str]:
    if kind is ColumnType.DATE:
        return [value.strftime(DATE_FORMAT) for value in pd.to_datetime(series)]
    if kind is ColumnType.INT:
        return ["" if pd.isna(value) else str(int(value)) for value in series]
    if kind is ColumnType.FLOAT:
        return [format_float(float(value), decimals) for value in series.astype("float64")]
    values = ["" if pd.isna(value) else str(value) for value in series]
    for value in values:
        if any(mark in value for mark in _FORBIDDEN):
            msg = "a text field contains a separator; the extract format has no quoting"
            raise ValueError(msg)
    return values


def to_csv_text(frame: pd.DataFrame, spec: ExtractSpec) -> str:
    """Render one typed frame exactly as the extract would be written."""
    types = COLUMN_TYPES[spec.name]
    decimals = 8 if spec.name is Extract.AGGREGATE else 2
    columns = [_format_column(frame[name], types[name], decimals) for name in spec.columns]
    rows = [",".join(values) for values in zip(*columns, strict=True)] if len(frame) else []
    lines: list[str] = [",".join(spec.columns)]
    if spec.name is Extract.AGGREGATE:
        empty = "," * (len(spec.columns) - 1)
        for row in rows:
            lines.extend((empty, row))
    else:
        lines.extend(rows)
    return LINE_TERMINATOR.join(lines) + LINE_TERMINATOR


@dataclass(frozen=True, slots=True)
class Marker:
    """Contents of ``SYNTHETIC.json``: what made these files, never when (it is deterministic)."""

    generator: str
    generator_version: str
    strata_version: str
    seed: int
    n_clients: int
    config_digest: str
    files: dict[str, str]


def _refuse_unsafe_destination(out_dir: Path, real_data_dir: Path | None) -> None:
    if real_data_dir is not None and is_within(out_dir, real_data_dir):
        msg = "Refusing to write synthetic files inside STRATA_DATA_DIR."
        raise PrivacyBoundaryError(msg)
    if out_dir.is_dir() and not (out_dir / SYNTHETIC_MARKER).is_file():
        present = [spec.filename for spec in EXTRACTS if (out_dir / spec.filename).exists()]
        if present:
            msg = "Refusing to overwrite extract files in a folder without the synthetic marker."
            raise PrivacyBoundaryError(msg)


def write_extract(
    extract: SyntheticExtract, out_dir: Path, *, real_data_dir: Path | None = None
) -> Marker:
    """Write the five files, the latent truth and the marker into ``out_dir``."""
    _refuse_unsafe_destination(out_dir, real_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for spec in EXTRACTS:
        payload = to_csv_text(extract.frames[spec.name], spec).encode(ENCODING)
        (out_dir / spec.filename).write_bytes(payload)
        hashes[spec.filename] = hashlib.sha256(payload).hexdigest()
    truth = extract.truth.to_csv(index=False, lineterminator="\n")
    (out_dir / TRUTH_FILE).write_text(truth, encoding="utf-8", newline="")
    marker = Marker(
        generator="strata.synthetic.parametric",
        generator_version=GENERATOR_VERSION,
        strata_version=__version__,
        seed=extract.config.seed,
        n_clients=extract.config.n_clients,
        config_digest=config_digest(extract.config),
        files=hashes,
    )
    (out_dir / SYNTHETIC_MARKER).write_text(
        json.dumps(asdict(marker), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline=""
    )
    return marker


def read_marker(data_dir: Path) -> Marker | None:
    """The synthetic marker of ``data_dir``, or ``None`` when the folder is not synthetic."""
    path = data_dir / SYNTHETIC_MARKER
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Marker(**payload)


def marker_matches(marker: Marker, data_dir: Path, names: Iterable[str]) -> bool:
    """True when every listed file still has the hash recorded at generation time."""
    for name in names:
        path = data_dir / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != marker.files.get(
            name
        ):
            return False
    return True
