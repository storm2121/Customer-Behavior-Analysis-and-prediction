"""Logs must never carry an identifier, even when a call site is careless."""

from __future__ import annotations

import json

import pytest

from strata.log import REDACTED, configure_logging, get_logger, mask_identifiers, redact_identifiers

FAKE_ID = "0f" * 20  # built at run time: no identifier-shaped literal lives in the source


def test_identifier_fields_are_replaced_whatever_they_hold() -> None:
    event = redact_identifiers(None, "info", {"event": "x", "client_id": 12, "identifier": "abc"})
    assert event["client_id"] == REDACTED
    assert event["identifier"] == REDACTED


def test_identifier_shaped_tokens_are_masked_inside_strings() -> None:
    event = redact_identifiers(None, "info", {"event": f"scored {FAKE_ID} today", "n": 3})
    assert FAKE_ID not in event["event"]
    assert REDACTED in event["event"]
    assert event["n"] == 3


def test_longer_hashes_are_left_alone() -> None:
    sha256 = "ab" * 32
    assert mask_identifiers(sha256) == sha256


def test_configured_logger_writes_redacted_json_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("DEBUG", json=True)
    get_logger("tests").info("loaded", client_id=FAKE_ID, note=f"row {FAKE_ID}")
    captured = capsys.readouterr()
    assert captured.out == ""
    record = json.loads(captured.err.strip().splitlines()[-1])
    assert record["client_id"] == REDACTED
    assert FAKE_ID not in captured.err
    assert record["module"] == "tests"


def test_level_filtering_applies(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", json=True)
    get_logger("tests").info("quiet")
    assert capsys.readouterr().err == ""
    configure_logging("NOT-A-LEVEL")
    get_logger("tests").info("console renderer")
    assert "console renderer" in capsys.readouterr().err
