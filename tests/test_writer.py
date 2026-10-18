"""The writer reproduces the extract's file format exactly, and refuses unsafe destinations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kashf.config import SYNTHETIC_MARKER, PrivacyBoundaryError
from kashf.contracts.specs import AGGREGATE, CARD, ENCODING, EXTRACTS, Extract
from kashf.ingest.readers import read_extract
from kashf.synthetic.parametric import ParametricConfig, SyntheticExtract, generate
from kashf.synthetic.writer import (
    TRUTH_FILE,
    format_float,
    marker_matches,
    read_marker,
    to_csv_text,
    write_extract,
)


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (1600.0, 2, "1600"),
        (20.5, 2, "20.5"),
        (133.95, 2, "133.95"),
        (0.01, 2, "0.01"),
        (0.0, 2, "0"),
        (89.0566935512, 8, "89.05669355"),
        (float("nan"), 2, ""),
    ],
)
def test_numbers_are_written_like_the_extract(value: float, decimals: int, expected: str) -> None:
    assert format_float(value, decimals) == expected


def test_files_are_latin1_crlf_with_a_final_line_break(synthetic_dir: Path) -> None:
    for spec in EXTRACTS:
        payload = (synthetic_dir / spec.filename).read_bytes()
        text = payload.decode(ENCODING)
        assert text.endswith("\r\n")
        assert text.count("\n") == text.count("\r\n")
        assert text.split("\r\n", 1)[0] == ",".join(spec.columns)
        assert '"' not in text


def test_accents_are_encoded_in_latin1(synthetic_dir: Path) -> None:
    header = (synthetic_dir / AGGREGATE.filename).read_bytes().split(b"\r\n", 1)[0]
    assert b"commer\xe7ant" in header


def test_aggregate_file_interleaves_empty_rows(synthetic_dir: Path) -> None:
    # Bytes, not read_text: text mode would translate CRLF and make this test pass vacuously.
    lines = (synthetic_dir / AGGREGATE.filename).read_bytes().decode(ENCODING).split("\r\n")[1:-1]
    assert len(lines) == 2 * 1000
    empty = "," * (len(AGGREGATE.columns) - 1)
    assert lines[0::2] == [empty] * (len(lines) // 2)
    assert all(line != empty for line in lines[1::2])


def test_dates_are_written_at_midnight(synthetic_dir: Path) -> None:
    second_line = (synthetic_dir / CARD.filename).read_bytes().decode(ENCODING).split("\r\n")[1]
    assert second_line.split(",")[1].endswith("T00:00")


def test_files_round_trip_through_the_reader(
    synthetic_dir: Path, synthetic_extract: SyntheticExtract
) -> None:
    for spec in EXTRACTS:
        frame = read_extract(synthetic_dir / spec.filename, spec).frame
        expected = synthetic_extract.frames[spec.name]
        if spec.name is Extract.AGGREGATE:
            frame = frame.dropna(subset=["identifier"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            frame, expected.reset_index(drop=True), check_exact=False, rtol=1e-8
        )


def test_marker_records_the_generator_and_every_file_hash(synthetic_dir: Path) -> None:
    marker = read_marker(synthetic_dir)
    assert marker is not None
    assert marker.seed == 42
    assert set(marker.files) == {spec.filename for spec in EXTRACTS}
    assert marker_matches(marker, synthetic_dir, marker.files)
    payload = json.loads((synthetic_dir / SYNTHETIC_MARKER).read_text(encoding="utf-8"))
    assert "created_at" not in payload  # deterministic output
    assert (synthetic_dir / TRUTH_FILE).is_file()


def test_writing_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    extract = generate(ParametricConfig(n_clients=50, seed=5))
    write_extract(extract, tmp_path / "a")
    write_extract(extract, tmp_path / "b")
    for spec in EXTRACTS:
        assert (tmp_path / "a" / spec.filename).read_bytes() == (
            tmp_path / "b" / spec.filename
        ).read_bytes()


def test_tampering_breaks_the_marker(tmp_path: Path) -> None:
    write_extract(generate(ParametricConfig(n_clients=30, seed=9)), tmp_path)
    marker = read_marker(tmp_path)
    assert marker is not None
    with (tmp_path / CARD.filename).open("ab") as handle:
        handle.write(b"x")
    assert not marker_matches(marker, tmp_path, marker.files)
    assert not marker_matches(marker, tmp_path, ["missing.csv"])


def test_no_marker_means_not_synthetic(tmp_path: Path) -> None:
    assert read_marker(tmp_path) is None


def test_writer_refuses_the_real_data_folder(tmp_path: Path) -> None:
    extract = generate(ParametricConfig(n_clients=20, seed=1))
    real = tmp_path / "extract"
    real.mkdir()
    with pytest.raises(PrivacyBoundaryError, match="KASHF_DATA_DIR"):
        write_extract(extract, real / "sub", real_data_dir=real)


def test_writer_refuses_to_overwrite_unmarked_extract_files(tmp_path: Path) -> None:
    (tmp_path / CARD.filename).write_text("not synthetic", encoding="utf-8")
    with pytest.raises(PrivacyBoundaryError, match="marker"):
        write_extract(generate(ParametricConfig(n_clients=20, seed=1)), tmp_path)


def test_text_fields_with_separators_are_refused() -> None:
    frame = generate(ParametricConfig(n_clients=20, seed=1)).frames[Extract.CARD].copy()
    frame.loc[frame.index[0], "type_origine"] = "Nat,ional"
    with pytest.raises(ValueError, match="separator"):
        to_csv_text(frame, CARD)


def test_an_empty_frame_writes_only_a_header() -> None:
    frame = generate(ParametricConfig(n_clients=20, seed=1)).frames[Extract.CARD].iloc[0:0]
    assert to_csv_text(frame, CARD) == ",".join(CARD.columns) + "\r\n"
