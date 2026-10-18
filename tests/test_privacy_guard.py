"""The privacy guard: every rule, the staged-content path, and the CLI exit codes.

Identifier-shaped strings are always built at run time (``"0f" * 20``), never written as literals,
so that this file itself passes the guard it tests.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from strata import privacy_guard as pg
from strata.cli import app
from strata.contracts.specs import AGGREGATE, BRANCH, CARD, CLIENTS, ENCODING, EXTRACTS, ExtractSpec
from tests.conftest import GitRunner

FAKE_ID = "0f" * 20
OTHER_ID = "a1" * 20
FALLBACK = pg.Scanner(matcher=pg.IdentifierMatcher.fallback())


def checks(path: str, content: str | bytes, scanner: pg.Scanner = FALLBACK) -> set[str]:
    data = content.encode("utf-8") if isinstance(content, str) else content
    return {finding.check for finding in scanner.scan(path, data)}


def write_fake_extract(folder: Path, identifiers: list[str]) -> None:
    """A miniature extract in a temporary folder, shaped like the real files."""
    folder.mkdir(parents=True, exist_ok=True)
    for spec in EXTRACTS:
        rows = [",".join(spec.columns)]
        for identifier in identifiers:
            values = [identifier] + ["1"] * (len(spec.columns) - 1)
            rows.append(",".join(values))
        (folder / spec.filename).write_text("\r\n".join(rows) + "\r\n", encoding=ENCODING)


# ----------------------------------------------------------------------------- identifier shape


def test_derived_pattern_matches_hex_identifiers_of_the_observed_length() -> None:
    pattern = pg.derive_identifier_pattern([FAKE_ID, OTHER_ID])
    assert pattern is not None
    assert pattern.search(f"id={FAKE_ID};")
    assert not pattern.search(FAKE_ID[:-1])
    assert not pattern.search("ab" * 32)  # a SHA-256 is longer, so it never matches


def test_short_identifiers_give_no_pattern() -> None:
    assert pg.derive_identifier_pattern(["1234", "5678"]) is None
    assert pg.derive_identifier_pattern([]) is None


def test_alphanumeric_and_punctuated_identifiers_are_described() -> None:
    alnum = pg.derive_identifier_pattern(["ZZ" + "9" * 12, "QQ" + "8" * 12])
    assert alnum is not None
    assert alnum.search("x ZZ999999999999 y")
    dashed = pg.derive_identifier_pattern(["abcd-1234-efgh-5678", "zzzz-0000-yyyy-1111"])
    assert dashed is not None
    assert dashed.search("(abcd-1234-efgh-5678)")


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.text(alphabet=string.hexdigits[:16], min_size=40, max_size=40), min_size=1, max_size=5
    )
)
def test_derived_pattern_finds_every_identifier_and_nothing_inside_longer_runs(
    identifiers: list[str],
) -> None:
    pattern = pg.derive_identifier_pattern(identifiers)
    assert pattern is not None
    for identifier in identifiers:
        assert pattern.search(f" {identifier},")
        assert not pattern.search(f"{identifier}{identifier}")


def test_matcher_learns_identifiers_from_a_data_dir(tmp_path: Path) -> None:
    write_fake_extract(tmp_path / "extract", [FAKE_ID])
    matcher = pg.IdentifierMatcher.from_data_dir(tmp_path / "extract")
    assert matcher.source == "derived"
    scanner = pg.Scanner(matcher=matcher)
    assert "identifier-exact" in checks("notes.md", f"seen {FAKE_ID} here", scanner)
    assert "identifier-format" in checks("notes.md", f"seen {OTHER_ID} here", scanner)


def test_matcher_falls_back_without_data(tmp_path: Path) -> None:
    assert pg.IdentifierMatcher.from_data_dir(None).source == "fallback"
    assert pg.IdentifierMatcher.from_data_dir(tmp_path / "missing").source == "fallback"
    write_fake_extract(tmp_path / "short", ["12345"])
    assert pg.IdentifierMatcher.from_data_dir(tmp_path / "short").source == "fallback"


def test_scanner_reads_the_environment(tmp_path: Path) -> None:
    write_fake_extract(tmp_path / "extract", [FAKE_ID])
    scanner = pg.Scanner.from_environment({"STRATA_DATA_DIR": str(tmp_path / "extract")})
    assert scanner.matcher.source == "derived"
    assert scanner.private_literals


# ----------------------------------------------------------------------------- rules


@pytest.mark.parametrize("spec", EXTRACTS, ids=lambda spec: spec.name.value)
def test_every_extract_filename_is_blocked(spec: ExtractSpec) -> None:
    assert "extract-filename" in checks(f"anywhere/{spec.filename}", "x")


@pytest.mark.parametrize("spec", EXTRACTS, ids=lambda spec: spec.name.value)
def test_every_extract_header_is_blocked(spec: ExtractSpec) -> None:
    assert "extract-header" in checks("dump.txt", ",".join(spec.columns) + "\nrow\n")


def test_a_partial_header_is_still_blocked() -> None:
    assert "extract-header" in checks("dump.txt", ",".join(CARD.columns[:3]))
    assert "extract-header" in checks("t.md", "| " + " | ".join(BRANCH.columns) + " |")


def test_header_matching_ignores_accents_quotes_and_case() -> None:
    header = ";".join(f'"{column.upper()}"' for column in AGGREGATE.columns).replace("Ç", "C")
    assert "extract-header" in checks("dump.txt", header)


def test_a_header_line_in_the_middle_of_a_file_is_not_a_header() -> None:
    text = '"""Docstring first."""\n' + ",".join(CLIENTS.columns)
    assert "extract-header" not in checks("module.py", text)


def test_csv_outside_the_allowed_folders_is_blocked() -> None:
    assert "csv-location" in checks("data/out.csv", "a,b\n1,2\n")
    assert "csv-location" not in checks("tests/fixtures/small.csv", "a,b\n1,2\n")
    assert "csv-location" not in checks("dbt/seeds/codes_x.csv", "a,b\n1,2\n")


def test_csv_size_limit_applies_everywhere() -> None:
    def table(rows: int) -> str:
        return "a,b\n" + "1,2\n" * rows

    assert "csv-size" not in checks("tests/fixtures/ok.csv", table(pg.MAX_CSV_ROWS))
    assert "csv-size" in checks("tests/fixtures/big.csv", table(pg.MAX_CSV_ROWS + 1))


@pytest.mark.parametrize(
    "name", ["w.duckdb", "x.parquet", "m.pkl", "s.xlsx", "claims.private.json"]
)
def test_data_files_are_blocked(name: str) -> None:
    assert "data-file" in checks(f"some/{name}", b"\x00\x01")


def test_identifiers_are_found_in_binary_content_too() -> None:
    blob = b"\x00\x00" + FAKE_ID.encode() + b"\x00"
    assert "identifier-format" in checks("blob.bin", blob)


def test_findings_never_contain_the_token() -> None:
    findings = FALLBACK.scan("notes.md", f"first line\nsecond {FAKE_ID}\n".encode())
    assert findings
    assert findings[0].line == 2
    for finding in findings:
        assert FAKE_ID not in finding.render()


def test_ordinary_text_is_clean() -> None:
    assert (
        checks("README.md", "# Title\n\nA sentence with 2024 and 1,000 and a sha " + "ab" * 32)
        == set()
    )


def ledger(claims: list[dict[str, object]]) -> str:
    return json.dumps({"schema_version": 1, "claims": claims})


def test_ledger_json_with_allow_listed_claims_passes() -> None:
    claim = {"id": "channel_mix", "n": 500, "value_n": {"share": 500}}
    assert "ledger-allowlist" not in checks("docs/ledger/claims.public.json", ledger([claim]))


@pytest.mark.parametrize(
    "claims",
    [
        [{"id": "silence_by_channel", "n": 500}],
        [{"id": "channel_mix", "n": 10}],
        [{"id": "channel_mix", "n": 500, "value_n": {"share": 3}}],
        [{"n": 500}],
    ],
    ids=["not-allow-listed", "small-n", "small-cell", "no-id"],
)
def test_ledger_json_violations_are_blocked(claims: list[dict[str, object]]) -> None:
    assert "ledger-allowlist" in checks("docs/ledger/claims.public.json", ledger(claims))


def test_ledger_json_shapes_are_checked() -> None:
    path = "docs/ledger/claims.public.json"
    assert "ledger-allowlist" in checks(path, "{not json")
    assert "ledger-allowlist" in checks(path, json.dumps({"claims": {"id": "x"}}))
    assert "ledger-allowlist" in checks(path, json.dumps([{"id": "secret_claim", "n": 99}]))
    assert "ledger-allowlist" not in checks(path, json.dumps("just a string"))


def test_private_path_literals_are_blocked_but_architect_documents_are_exempt(
    tmp_path: Path,
) -> None:
    private = tmp_path / "SecretPlace"
    scanner = pg.Scanner.from_environment({"STRATA_DATA_DIR": str(private)})
    spelled = str(private).replace("\\", "/")
    assert "private-path" in checks("strata/x.py", f'PATH = "{spelled}"', scanner)
    assert "private-path" in checks("docs/x.md", str(private).upper(), scanner)
    assert "private-path" not in checks("ARCHITECTURE.md", spelled, scanner)


def test_notebooks_with_outputs_are_blocked() -> None:
    dirty = {"cells": [{"cell_type": "code", "execution_count": 3, "outputs": [{"text": "x"}]}]}
    clean = {"cells": [{"cell_type": "code", "execution_count": None, "outputs": []}]}
    assert "notebook-outputs" in checks("notebooks/a.ipynb", json.dumps(dirty))
    assert "notebook-outputs" not in checks("notebooks/a.ipynb", json.dumps(clean))
    assert "notebook-outputs" in checks("notebooks/a.ipynb", "{broken")


# ----------------------------------------------------------------------------- git integration


def test_staged_fixture_mimicking_an_extract_header_is_blocked(
    git_repo: Path, git: GitRunner
) -> None:
    """The M0 definition of done: a staged extract-like fixture cannot be committed."""
    leak = git_repo / "notes" / "sample.csv"
    leak.parent.mkdir()
    leak.write_text(",".join(CARD.columns) + "\r\nX,2024-01-01T00:00,10,WITHDRAWAL,National\r\n")
    git("add", "notes/sample.csv")
    found = {f.check for f in pg.scan(pg.staged_files(git_repo), FALLBACK)}
    assert {"extract-header", "csv-location"} <= found


def test_the_staged_content_is_scanned_not_the_working_tree(git_repo: Path, git: GitRunner) -> None:
    target = git_repo / "notes.md"
    target.write_text(f"leak {FAKE_ID}\n")
    git("add", "notes.md")
    target.write_text("cleaned in the working tree only\n")
    found = {f.check for f in pg.scan(pg.staged_files(git_repo), FALLBACK)}
    assert "identifier-format" in found


def test_tracked_and_explicit_files_are_scanned(
    git_repo: Path, git: GitRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    (git_repo / "ok.md").write_text("fine\n")
    (git_repo / "bad.md").write_text(f"{FAKE_ID}\n")
    git("add", "ok.md", "bad.md")
    git("commit", "-q", "-m", "seed")
    (git_repo / "ok.md").unlink()
    tracked = dict(pg.tracked_files(git_repo))
    assert set(tracked) == {"bad.md"}
    monkeypatch.chdir(git_repo)
    explicit = dict(pg.disk_files([Path("bad.md"), Path("missing.md")], git_repo))
    assert set(explicit) == {"bad.md"}
    outside = dict(pg.disk_files([git_repo.parent / "conftest-absent.md"], git_repo))
    assert outside == {}


def test_cli_blocks_staged_leaks_with_exit_code_one(
    git_repo: Path, git: GitRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRATA_REPO_ROOT", str(git_repo))
    (git_repo / "dump.txt").write_text(",".join(CLIENTS.columns) + f"\n{FAKE_ID},1\n")
    git("add", "dump.txt")
    result = CliRunner().invoke(app, ["privacy-check", "--staged"])
    assert result.exit_code == 1
    assert "BLOCKED" in result.output
    assert "extract-header" in result.output
    assert FAKE_ID not in result.output


def test_cli_reports_clean_trees_and_usage_errors(
    git_repo: Path, git: GitRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRATA_REPO_ROOT", str(git_repo))
    (git_repo / "fine.md").write_text("nothing to see\n")
    git("add", "fine.md")
    runner = CliRunner()
    assert runner.invoke(app, ["privacy-check", "--staged"]).exit_code == 0
    assert runner.invoke(app, ["privacy-check", "--all-files"]).exit_code == 0
    monkeypatch.chdir(git_repo)
    assert runner.invoke(app, ["privacy-check", "fine.md"]).exit_code == 0
    assert runner.invoke(app, ["privacy-check"]).exit_code == 2
