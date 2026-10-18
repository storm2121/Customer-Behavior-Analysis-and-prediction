"""Pre-commit scanner that keeps the real extract, and anything shaped like it, out of git.

The guard is deliberately paranoid and deliberately quiet: it blocks on suspicion, and it never
echoes what it found. A finding names the file, the line and the rule; the offending token stays
where it is, so that running the guard cannot itself leak an identifier into a terminal or a CI log.

Rules (ids appear in every finding):

``extract-filename``
    A file named like one of the five extracts.
``extract-header``
    A text file whose first line looks like an extract header (most of one extract's columns).
``csv-location`` / ``csv-size``
    A CSV outside ``tests/fixtures/`` and ``dbt/seeds/``, or any CSV with more than 200 rows.
``data-file``
    A binary data format (DuckDB, Parquet, pickles, spreadsheets, ...) or a private claims file.
``identifier-exact`` / ``identifier-format``
    A token equal to a real identifier, or shaped like one. The shape is derived at run time from
    the extract when ``STRATA_DATA_DIR`` is available, and falls back to a conservative pattern
    (32 to 40 hexadecimal characters) when it is not, which is the case in CI.
``ledger-allowlist``
    A JSON file under ``docs/ledger/`` holding a claim id outside the allowlist, a claim computed on
    fewer than the minimum number of clients, or a file that cannot be parsed.
``private-path``
    The literal location of the extract or of the private outputs, read from the environment.
``notebook-outputs``
    A notebook with outputs or execution counts (outputs can carry data).
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import re
import string
import subprocess
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from strata.contracts.specs import ENCODING, EXTRACTS
from strata.ledger.allowlist import public_violations

MAX_CSV_ROWS = 200
CSV_ALLOWED_PREFIXES = ("tests/fixtures/", "dbt/seeds/")
CSV_SUFFIXES = frozenset({".csv", ".tsv"})
DATA_SUFFIXES = frozenset(
    {
        ".duckdb",
        ".parquet",
        ".feather",
        ".arrow",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".xlsx",
        ".xls",
        ".pkl",
        ".pickle",
        ".joblib",
        ".sav",
        ".dta",
        ".npz",
    }
)
DATA_NAMES = frozenset({"claims.private.json"})
LEDGER_PREFIX = "docs/ledger/"

ARCHITECT_DOCUMENTS = frozenset({"AGENT_RULES.md", "ARCHITECTURE.md", "MILESTONES.md"})
"""Governance documents owned by the architect session. They predate the guard and describe the
machine they were written on, so the ``private-path`` rule skips them; every other rule applies."""

HEADER_MIN_MATCHES = 3
HEADER_MIN_RATIO = 0.6
MIN_DERIVED_LENGTH = 12
"""Identifiers shorter than this make a format regex useless (it would match ordinary numbers)."""

FALLBACK_IDENTIFIER = re.compile(r"(?<![0-9A-Za-z])[0-9A-Fa-f]{32,40}(?![0-9A-Za-z])")

_EXTRACT_STEMS = tuple(
    spec.filename.lower().removesuffix(".csv").removesuffix("(in)") for spec in EXTRACTS
)


@dataclass(frozen=True, order=True, slots=True)
class Finding:
    """One reason to block a file. Never carries the offending content."""

    path: str
    line: int
    check: str
    message: str

    def render(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        return f"{location}: [{self.check}] {self.message}"


@dataclass(frozen=True, slots=True)
class IdentifierMatcher:
    """Recognises identifiers without ever exposing them.

    ``known`` holds the real identifiers (lower-cased) in memory for exact matching; it is built
    from ``STRATA_DATA_DIR`` on the machine that has the data and is empty everywhere else.
    """

    patterns: tuple[re.Pattern[str], ...]
    known: frozenset[str] = field(default_factory=frozenset)
    source: str = "fallback"

    @classmethod
    def fallback(cls) -> IdentifierMatcher:
        return cls(patterns=(FALLBACK_IDENTIFIER,))

    @classmethod
    def from_data_dir(cls, data_dir: Path | None) -> IdentifierMatcher:
        """Learn the identifier shape from the extract; fall back when it is unavailable."""
        if data_dir is None or not data_dir.is_dir():
            return cls.fallback()
        identifiers = set(read_identifiers(data_dir))
        derived = derive_identifier_pattern(identifiers)
        if derived is None:
            return cls.fallback()
        return cls(
            patterns=(derived, FALLBACK_IDENTIFIER),
            known=frozenset(value.lower() for value in identifiers),
            source="derived",
        )

    def scan(self, text: str) -> Iterator[tuple[int, bool]]:
        """Yield ``(line_number, is_exact)`` for every identifier-like token in ``text``."""
        for number, line in enumerate(text.splitlines(), start=1):
            seen: set[tuple[int, int]] = set()
            for pattern in self.patterns:
                for match in pattern.finditer(line):
                    if match.span() in seen:
                        continue
                    seen.add(match.span())
                    yield number, match.group(0).lower() in self.known


def read_identifiers(data_dir: Path) -> Iterator[str]:
    """Identifier values from every extract present in ``data_dir`` (held in memory only)."""
    for spec in EXTRACTS:
        path = data_dir / spec.filename
        if not path.is_file():
            continue
        with path.open(encoding=ENCODING, newline="") as handle:
            for row in csv.DictReader(handle):
                value = (row.get(spec.id_column) or "").strip()
                if value:
                    yield value


def derive_identifier_pattern(identifiers: Iterable[str]) -> re.Pattern[str] | None:
    """A regex describing the identifiers' alphabet and length, or ``None`` if too generic.

    The regex is bounded by non-alphanumeric context so that it never fires inside a longer token
    (a SHA-256 checksum in a lock file, for example).
    """
    values = [value for value in identifiers if value]
    if not values:
        return None
    lengths = {len(value) for value in values}
    shortest, longest = min(lengths), max(lengths)
    if shortest < MIN_DERIVED_LENGTH:
        return None
    alphabet = set("".join(values))
    if alphabet <= set(string.hexdigits):
        char_class = "0-9A-Fa-f"
    elif alphabet <= set(string.ascii_letters + string.digits):
        char_class = "0-9A-Za-z"
    else:
        extras = "".join(sorted(alphabet - set(string.ascii_letters + string.digits)))
        char_class = "0-9A-Za-z" + re.escape(extras)
    quantifier = f"{{{shortest}}}" if shortest == longest else f"{{{shortest},{longest}}}"
    boundary = "0-9A-Za-z"
    return re.compile(f"(?<![{boundary}])[{char_class}]{quantifier}(?![{boundary}])")


def _normalise_token(token: str) -> str:
    folded = unicodedata.normalize("NFKD", token.strip().strip("\"'`|").strip())
    return "".join(ch for ch in folded if not unicodedata.combining(ch)).lower()


_EXTRACT_COLUMN_SETS = tuple(
    (spec, frozenset(_normalise_token(column) for column in spec.columns)) for spec in EXTRACTS
)


def header_matches(line: str) -> str | None:
    """The extract a header line imitates, if any (by column overlap, accent-insensitive)."""
    tokens = {_normalise_token(token) for token in re.split(r"[,;\t|]", line) if token.strip()}
    for spec, columns in _EXTRACT_COLUMN_SETS:
        overlap = len(tokens & columns)
        needed = min(HEADER_MIN_MATCHES, len(columns))
        if overlap >= needed and overlap / len(columns) >= HEADER_MIN_RATIO:
            return spec.name.value
    return None


def _decode(data: bytes) -> tuple[str, bool]:
    """Text of ``data`` and whether it looked binary (binary data is still scanned as Latin-1)."""
    binary = b"\x00" in data[:8192]
    if not binary:
        try:
            return data.decode("utf-8-sig"), False
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1"), binary


def _first_line(text: str) -> tuple[int, str] | None:
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            return number, line
    return None


def _private_literals(environ: dict[str, str]) -> list[str]:
    """Spellings of the private locations, as they could appear in a file (lower-cased)."""
    literals: set[str] = set()
    for variable in ("STRATA_DATA_DIR", "STRATA_PRIVATE_DIR"):
        raw = environ.get(variable, "").strip().strip('"')
        if len(raw) < 4:
            continue
        spellings = {raw, raw.rstrip("/\\")}
        with contextlib.suppress(OSError):
            spellings.add(str(Path(raw).expanduser().resolve()))
        variants = {
            variant
            for spelling in spellings
            for variant in (
                spelling,
                spelling.replace("\\", "/"),
                spelling.replace("/", "\\"),
                spelling.replace("\\", "\\\\"),
            )
        }
        literals.update(value.lower() for value in variants if len(value) >= 4)
    return sorted(literals, key=len, reverse=True)


def _claim_records(payload: object) -> list[object] | None:
    """The list of claim records in a ledger file, or ``None`` if the shape is wrong."""
    records: object = payload.get("claims", []) if isinstance(payload, dict) else payload
    if isinstance(records, list):
        return records
    if isinstance(payload, dict):
        return None
    return []


def _ledger_findings(path: str, text: str) -> list[Finding]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [Finding(path, 0, "ledger-allowlist", "ledger JSON does not parse; cannot verify")]
    records = _claim_records(payload)
    if records is None:
        return [Finding(path, 0, "ledger-allowlist", "ledger JSON has no list of claims")]
    return [Finding(path, 0, "ledger-allowlist", problem) for problem in public_violations(records)]


def _notebook_findings(path: str, text: str) -> list[Finding]:
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return [Finding(path, 0, "notebook-outputs", "notebook does not parse; cannot verify")]
    cells = notebook.get("cells", []) if isinstance(notebook, dict) else []
    dirty = any(
        isinstance(cell, dict) and (cell.get("outputs") or cell.get("execution_count") is not None)
        for cell in cells
    )
    if dirty:
        return [Finding(path, 0, "notebook-outputs", "notebook has outputs; run nbstripout")]
    return []


def _name_findings(rel: str, name: str, suffix: str) -> list[Finding]:
    findings: list[Finding] = []
    if any(name.startswith(stem) for stem in _EXTRACT_STEMS):
        findings.append(Finding(rel, 0, "extract-filename", "file named like a bank extract"))
    if suffix in DATA_SUFFIXES or name in DATA_NAMES:
        findings.append(Finding(rel, 0, "data-file", "data file format that may hold rows"))
    return findings


def _header_findings(rel: str, text: str) -> list[Finding]:
    first = _first_line(text)
    if first is None:
        return []
    imitated = header_matches(first[1])
    if imitated is None:
        return []
    return [Finding(rel, first[0], "extract-header", f"header imitates {imitated}")]


def _csv_findings(rel: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    if not rel.startswith(CSV_ALLOWED_PREFIXES):
        findings.append(
            Finding(rel, 0, "csv-location", "CSV outside tests/fixtures/ and dbt/seeds/")
        )
    rows = sum(1 for row in io.StringIO(text) if row.strip()) - 1
    if rows > MAX_CSV_ROWS:
        findings.append(Finding(rel, 0, "csv-size", f"CSV has more than {MAX_CSV_ROWS} rows"))
    return findings


@dataclass(slots=True)
class Scanner:
    """Applies every rule to one file's bytes. Construct once per run (it learns identifiers)."""

    matcher: IdentifierMatcher
    private_literals: list[str] = field(default_factory=list)

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> Scanner:
        env = dict(os.environ if environ is None else environ)
        data_dir = env.get("STRATA_DATA_DIR")
        matcher = IdentifierMatcher.from_data_dir(Path(data_dir) if data_dir else None)
        return cls(matcher=matcher, private_literals=_private_literals(env))

    def scan(self, path: str, data: bytes) -> list[Finding]:
        posix = PurePosixPath(path.replace("\\", "/"))
        rel = posix.as_posix()
        suffix = posix.suffix.lower()
        text, binary = _decode(data)

        findings = _name_findings(rel, posix.name.lower(), suffix)
        if not binary:
            findings.extend(_header_findings(rel, text))
        if suffix in CSV_SUFFIXES:
            findings.extend(_csv_findings(rel, text))
        findings.extend(self._identifier_findings(rel, text))
        if rel.startswith(LEDGER_PREFIX) and suffix == ".json":
            findings.extend(_ledger_findings(rel, text))
        if suffix == ".ipynb":
            findings.extend(_notebook_findings(rel, text))
        if self.private_literals and posix.name not in ARCHITECT_DOCUMENTS:
            findings.extend(self._private_path_findings(rel, text))
        return findings

    def _identifier_findings(self, rel: str, text: str) -> list[Finding]:
        return [
            Finding(rel, line, "identifier-exact", "token equal to a real identifier")
            if exact
            else Finding(rel, line, "identifier-format", "token shaped like an identifier")
            for line, exact in self.matcher.scan(text)
        ]

    def _private_path_findings(self, rel: str, text: str) -> list[Finding]:
        for number, content in enumerate(text.lower().splitlines(), start=1):
            if any(literal in content for literal in self.private_literals):
                return [Finding(rel, number, "private-path", "hard-coded private location")]
        return []


def _git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    return completed.stdout


def staged_files(repo: Path) -> Iterator[tuple[str, bytes]]:
    """``(path, staged content)`` for every added, copied, modified or renamed file in the index."""
    names = _git(repo, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRT")
    for raw in names.split(b"\x00"):
        if raw:
            path = raw.decode("utf-8")
            yield path, _git(repo, "show", f":{path}")


def tracked_files(repo: Path) -> Iterator[tuple[str, bytes]]:
    """``(path, working-tree content)`` for every tracked file that still exists."""
    for raw in _git(repo, "ls-files", "-z").split(b"\x00"):
        if not raw:
            continue
        path = raw.decode("utf-8")
        file = repo / path
        if file.is_file():
            yield path, file.read_bytes()


def disk_files(paths: Sequence[Path], repo: Path) -> Iterator[tuple[str, bytes]]:
    """``(repo-relative path, content)`` for explicit files, as pre-commit passes them."""
    for path in paths:
        absolute = path if path.is_absolute() else Path.cwd() / path
        if not absolute.is_file():
            continue
        try:
            rel = absolute.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            rel = absolute.as_posix()
        yield rel, absolute.read_bytes()


def scan(files: Iterable[tuple[str, bytes]], scanner: Scanner | None = None) -> list[Finding]:
    """Run every rule over ``files`` and return the findings sorted by location."""
    active = scanner or Scanner.from_environment()
    findings: list[Finding] = []
    for path, data in files:
        findings.extend(active.scan(path, data))
    return sorted(set(findings))
