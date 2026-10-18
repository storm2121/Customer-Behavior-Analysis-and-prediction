"""Source mappings: the reference instance reproduced; an alien schema gives the same marts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from pydantic import ValidationError

from strata.config import Settings, Zone
from strata.contracts.extracts import ColumnType, ContractError
from strata.contracts.mapping import (
    CLIENT_TYPES,
    EventFile,
    SourceMapping,
    canonical_events,
    load_mapping,
    mapping_path,
)
from strata.contracts.specs import BY_NAME, Extract
from strata.dbt_runner import run_dbt
from strata.ingest import dq
from strata.ingest.load import ingest
from strata.ingest.readers import read_delimited
from strata.synthetic.writer import read_marker
from strata.warehouse import connect, read_relation, read_table
from tests.alien import AlienSchema, alien_schemas, write_alien_source
from tests.conftest import public_settings
from tests.reference_snapshot import FIXTURE, snapshot


def test_the_reference_mappings_follow_the_contracts() -> None:
    reference = load_mapping("reference_cih").mapping
    synthetic = load_mapping("synthetic").mapping
    # The generator writes the reference format exactly: the two mappings differ only in name.
    assert reference.model_dump(exclude={"source", "description"}) == synthetic.model_dump(
        exclude={"source", "description"}
    )
    for file in [reference.clients, reference.client_aggregates, *reference.events]:
        assert file is not None
        assert file.contract is not None
        spec = BY_NAME[file.contract]
        assert file.file == spec.filename
        assert set(file.columns.values()) <= set(spec.columns)
    assert set(reference.clients.columns) == set(CLIENT_TYPES)


def test_mapping_names_resolve_to_files(tmp_path: Path) -> None:
    # Not in the given root: the checkout's own mappings are used.
    assert mapping_path("synthetic", tmp_path).is_file()
    (tmp_path / "mappings").mkdir()
    (tmp_path / "mappings" / "synthetic.yaml").write_text("x", encoding="utf-8")
    assert mapping_path("synthetic", tmp_path) == tmp_path / "mappings" / "synthetic.yaml"
    assert mapping_path(tmp_path / "x.yaml") == tmp_path / "x.yaml"
    loaded = load_mapping("synthetic")
    assert loaded.path == "synthetic.yaml"
    assert len(loaded.sha256) == 64


def _event(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "file": "e.csv",
        "channel": "card",
        "columns": {"client_id": "c", "ts": "t", "amount": "a"},
    }
    return {**base, **changes}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"channel_column": "ch", "channel_codes": {"C": "card"}}, "not both"),
        ({"channel": None, "channel_column": "ch"}, "needs `channel_codes`"),
        ({"columns": {"client_id": "c", "ts": "t", "colour": "x"}}, "not canonical event"),
        ({"columns": {"client_id": "c"}}, "needs a `ts`"),
        ({"channel": "teleport"}, "Input should be"),
    ],
)
def test_event_mappings_are_validated(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EventFile.model_validate(_event(**changes))


def test_client_mappings_need_canonical_columns_and_an_id() -> None:
    base = {"source": "s", "description": "d", "events": [_event()]}
    with pytest.raises(ValidationError, match="not canonical columns"):
        SourceMapping.model_validate({**base, "clients": {"file": "c", "columns": {"x": "y"}}})
    with pytest.raises(ValidationError, match="needs a `client_id`"):
        SourceMapping.model_validate({**base, "clients": {"file": "c", "columns": {"age": "y"}}})
    with pytest.raises(ValidationError, match="extra"):
        SourceMapping.model_validate({**base, "clients": {"file": "c", "columns": {}}, "x": 1})


def test_unmapped_codes_stop_the_ingest() -> None:
    file = EventFile.model_validate(
        _event(columns={"client_id": "c", "ts": "t", "kind": "k"}, kind_codes={"P": "ACHAT"})
    )
    frame = pd.DataFrame(
        {"c": ["a", "b"], "t": pd.to_datetime(["2024-01-01"] * 2), "k": ["P", "Z"]}
    )
    with pytest.raises(ContractError):
        canonical_events(frame, file, label="events")
    several = EventFile.model_validate(
        _event(channel=None, channel_column="ch", channel_codes={"1": "card"})
    )
    frame = pd.DataFrame({"c": ["a"], "t": pd.to_datetime(["2024-01-01"]), "a": [1.0], "ch": ["9"]})
    with pytest.raises(ContractError):
        canonical_events(frame, several, label="events")


def test_generic_reading_parses_the_source_format(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_bytes("id;day;value\nA;31/12/2023;1,5\nB;01/01/2024;\n".encode("latin-1"))
    read = read_delimited(
        path,
        label="events",
        columns={"id": ColumnType.ID, "day": ColumnType.DATE, "value": ColumnType.FLOAT},
        encoding="latin-1",
        delimiter=";",
        decimal=",",
        date_format="%d/%m/%Y",
    )
    assert read.frame["value"].iloc[0] == 1.5
    assert pd.isna(read.frame["value"].iloc[1])
    assert read.frame["day"].iloc[0] == pd.Timestamp("2023-12-31")
    with pytest.raises(ContractError):
        read_delimited(
            path,
            label="events",
            columns={"missing": ColumnType.FLOAT},
            encoding="latin-1",
            delimiter=";",
            decimal=",",
            date_format="%d/%m/%Y",
        )
    with pytest.raises(ContractError):
        read_delimited(
            path,
            label="events",
            columns={"day": ColumnType.DATE},
            encoding="latin-1",
            delimiter=";",
            decimal=",",
            date_format="%Y-%m-%d",
        )


def test_the_reference_mapping_reproduces_the_pipeline_before_mappings(
    built_warehouse: Settings,
) -> None:
    """DoD of M4.5: same manifest, same data-quality report, same row count in every model."""
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual = json.loads(json.dumps(snapshot(built_warehouse), default=str))
    assert actual["fingerprint"] == expected["fingerprint"]
    assert actual["files"] == expected["files"]
    assert actual["dq_report"] == expected["dq_report"]
    assert actual["row_counts"] == expected["row_counts"]


def test_the_canonical_layer_holds_every_event_of_every_file(built_warehouse: Settings) -> None:
    con = connect(built_warehouse.workspace().warehouse, read_only=True)
    try:
        events = read_table(con, "raw", "events")
        manifest = read_table(con, "raw", "manifest")
    finally:
        con.close()
    assert set(events["channel"]) == {"card", "online", "branch"}
    rows = manifest.set_index("extract")["rows"]
    for extract, channel in (
        (Extract.CARD, "card"),
        (Extract.ONLINE, "online"),
        (Extract.BRANCH, "branch"),
    ):
        assert (events["channel"] == channel).sum() == rows[extract.value]
    assert set(manifest["role"]) == {"clients", "events", "client_aggregates"}


def _canonical(settings: Settings) -> dict[str, pd.DataFrame]:
    con = connect(settings.workspace().warehouse, read_only=True)
    try:
        return {
            name: read_relation(con, f"raw.{name}")
            for name in ("clients", "events", "client_aggregates")
        }
    finally:
        con.close()


def _marts(settings: Settings) -> dict[str, pd.DataFrame]:
    con = connect(settings.workspace().warehouse, read_only=True)
    try:
        names = [
            row[0]
            for row in con.execute(
                "select table_name from information_schema.tables where table_schema = 'marts'"
            ).fetchall()
        ]
        return {name: read_relation(con, f"marts.{name}") for name in sorted(names)}
    finally:
        con.close()


EVENT_KEYS = ["channel", "client_id", "ts", "kind", "amount", "direction", "origin", "currency"]


def _sorted_events(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Events in a canonical order, without the file and row they came from."""
    events = tables["events"][EVENT_KEYS]
    return events.sort_values(EVENT_KEYS, na_position="last").reset_index(drop=True)


def _assert_same(left: pd.DataFrame, right: pd.DataFrame, name: str) -> None:
    assert list(left.columns) == list(right.columns), name
    assert len(left) == len(right), name
    for column in left.columns:
        a, b = left[column], right[column]
        if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
            np.testing.assert_allclose(
                a.to_numpy(dtype=float),
                b.to_numpy(dtype=float),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"{name}.{column}",
            )
        else:
            assert (
                a.astype("object").where(a.notna(), None).tolist()
                == b.astype("object").where(b.notna(), None).tolist()
            ), f"{name}.{column}"


@pytest.mark.slow
@settings(
    max_examples=3,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(schema=alien_schemas())
def test_an_alien_schema_maps_onto_the_same_marts(
    schema: AlienSchema,
    built_warehouse: Settings,
    synthetic_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """DoD of M4.5: a generated, deliberately different schema gives identical marts."""
    reference_tables = _canonical(built_warehouse)
    folder = tmp_path_factory.mktemp("alien")
    base = read_marker(synthetic_dir)
    assert base is not None
    mapping = write_alien_source(reference_tables, folder / "data", schema, base)
    alien = public_settings(folder / "work")
    report = ingest(alien, Zone.PUBLIC, folder / "data", mapping)
    assert report.ok
    assert report.manifest.mapping == "alien"
    con = connect(alien.workspace().warehouse)
    try:
        results = dq.run(con)
    finally:
        con.close()
    assert not dq.has_errors(results)
    alien_tables = _canonical(alien)
    for name in ("clients", "client_aggregates"):
        _assert_same(reference_tables[name], alien_tables[name], name)
    _assert_same(_sorted_events(reference_tables), _sorted_events(alien_tables), "events")
    assert run_dbt(alien, Zone.PUBLIC, ["build"]).success
    reference_marts, alien_marts = _marts(built_warehouse), _marts(alien)
    assert reference_marts.keys() == alien_marts.keys()
    for name in reference_marts:
        _assert_same(reference_marts[name], alien_marts[name], name)
    assert yaml.safe_load(mapping.read_text(encoding="utf-8"))["source"] == "alien"
