"""A deliberately different source schema, drawn by hypothesis, built from the synthetic data.

The portability test re-shapes the canonical tables of the reference warehouse into files that share
nothing with the reference extract's format: other column names, another delimiter and decimal
mark, another date format and encoding, recoded channels, kinds and directions, shuffled rows, and
either one event file per channel or a single file with a channel column. It then writes the mapping
that describes them. If the pipeline is source-agnostic, ingesting those files through that mapping
must give the same marts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hypothesis import strategies as st

from strata.contracts.mapping import AGGREGATE_TYPES, CHANNELS, CLIENT_TYPES, EVENT_TYPES
from strata.synthetic.writer import Marker, write_derived_marker


@dataclass(frozen=True)
class AlienSchema:
    prefix: str
    upper: bool
    delimiter: str
    decimal: str
    date_format: str
    encoding: str
    split_events: bool
    kind_tag: str
    channel_codes: tuple[str, str, str]
    direction_codes: tuple[str, str]
    shuffle_seed: int

    def column(self, name: str) -> str:
        alien = f"{self.prefix}{name}"
        return alien.upper() if self.upper else alien


@st.composite
def alien_schemas(draw: st.DrawFn) -> AlienSchema:
    delimiter = draw(st.sampled_from([";", "|", "\t", ","]))
    decimal = "." if delimiter == "," else draw(st.sampled_from([".", ","]))
    return AlienSchema(
        prefix=draw(st.sampled_from(["x_", "col_", "src_", "f"])),
        upper=draw(st.booleans()),
        delimiter=delimiter,
        decimal=decimal,
        date_format=draw(st.sampled_from(["%Y-%m-%d", "%d/%m/%Y", "%Y%m%d", "%d.%m.%Y %H:%M"])),
        encoding=draw(st.sampled_from(["utf-8", "latin-1"])),
        split_events=draw(st.booleans()),
        kind_tag=draw(st.sampled_from(["K", "type-", "cat_"])),
        channel_codes=draw(
            st.sampled_from([("C", "O", "B"), ("1", "2", "3"), ("crd", "web", "agc")])
        ),
        direction_codes=draw(st.sampled_from([("+", "-"), ("CR", "DB"), ("in", "out")])),
        shuffle_seed=draw(st.integers(min_value=0, max_value=2**16)),
    )


def _text(series: pd.Series, decimal: str, date_format: str) -> pd.Series:
    """A column as the alien source writes it (empty for missing)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime(date_format).fillna("")
    if pd.api.types.is_float_dtype(series):
        return series.map(lambda v: "" if pd.isna(v) else repr(float(v)).replace(".", decimal))
    if pd.api.types.is_integer_dtype(series):
        return series.map(lambda v: "" if pd.isna(v) else str(int(v)))
    return series.astype("object").map(lambda v: "" if v is None or pd.isna(v) else str(v))


def _write(frame: pd.DataFrame, path: Path, schema: AlienSchema) -> None:
    text = frame.to_csv(index=False, sep=schema.delimiter, lineterminator="\n")
    path.write_bytes(text.encode(schema.encoding))


def write_alien_source(
    tables: dict[str, pd.DataFrame], out_dir: Path, schema: AlienSchema, base: Marker
) -> Path:
    """Write the alien files, their synthetic marker and their mapping; return the mapping path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rename = {name: schema.column(name) for name in {*CLIENT_TYPES, *AGGREGATE_TYPES, *EVENT_TYPES}}
    files: list[str] = []

    clients = tables["clients"][list(CLIENT_TYPES)]
    clients_out = pd.DataFrame(
        {rename[c]: _text(clients[c], schema.decimal, schema.date_format) for c in clients}
    )
    _write(clients_out, out_dir / "people.txt", schema)
    files.append("people.txt")

    aggregates = tables["client_aggregates"][list(AGGREGATE_TYPES)]
    aggregates_out = pd.DataFrame(
        {rename[c]: _text(aggregates[c], schema.decimal, schema.date_format) for c in aggregates}
    )
    _write(aggregates_out, out_dir / "summary.txt", schema)
    files.append("summary.txt")

    events = tables["events"].copy()
    kinds = sorted(events["kind"].dropna().unique().tolist())
    kind_codes = {f"{schema.kind_tag}{index}": kind for index, kind in enumerate(kinds)}
    to_alien_kind = {kind: code for code, kind in kind_codes.items()}
    channel_codes = dict(zip(schema.channel_codes, CHANNELS, strict=True))
    to_alien_channel = {channel: code for code, channel in channel_codes.items()}
    direction_codes = dict(zip(schema.direction_codes, ("C", "D"), strict=True))
    to_alien_direction = {value: code for code, value in direction_codes.items()}
    events["kind"] = events["kind"].map(to_alien_kind)
    events["direction"] = events["direction"].map(to_alien_direction)
    events["channel"] = events["channel"].map(to_alien_channel)
    order = np.random.default_rng(schema.shuffle_seed).permutation(len(events))
    events = events.iloc[order].reset_index(drop=True)
    event_columns = [c for c in EVENT_TYPES if c != "channel"]
    mapping_events: list[dict[str, object]] = []
    groups = (
        [(channel, events[events["channel"] == to_alien_channel[channel]]) for channel in CHANNELS]
        if schema.split_events
        else [(None, events)]
    )
    for channel, group in groups:
        name = f"events_{channel}.txt" if channel else "events.txt"
        columns = event_columns if channel else [*event_columns, "channel"]
        out = pd.DataFrame(
            {rename[c]: _text(group[c], schema.decimal, schema.date_format) for c in columns}
        )
        _write(out, out_dir / name, schema)
        files.append(name)
        entry: dict[str, object] = {
            "file": name,
            "columns": {c: rename[c] for c in event_columns},
            "kind_codes": kind_codes,
            "direction_codes": direction_codes,
        }
        if channel:
            entry["channel"] = channel
        else:
            entry["channel_column"] = rename["channel"]
            entry["channel_codes"] = channel_codes
        mapping_events.append(entry)

    mapping = {
        "source": "alien",
        "description": "A generated schema that shares nothing with the reference format.",
        "encoding": schema.encoding,
        "delimiter": schema.delimiter,
        "decimal": schema.decimal,
        "date_format": schema.date_format,
        "clients": {"file": "people.txt", "columns": {c: rename[c] for c in CLIENT_TYPES}},
        "events": mapping_events,
        "client_aggregates": {
            "file": "summary.txt",
            "columns": {c: rename[c] for c in AGGREGATE_TYPES},
        },
    }
    path = out_dir / "alien.yaml"
    path.write_text(yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_derived_marker(out_dir, files, base, generator="tests.alien")
    return path
