"""``kashf ledger check``: the documents say exactly what the ledger says, and nothing else.

Four rules, each a CI failure:

1. ``docs/ledger/claims.public.json`` holds only allow-listed claims within the publication rules.
2. Every ``docs/findings/*.md`` is the current rendering of its template (no drift, no hand edits),
   and every findings document has a template.
3. The README's ledger-governed section is the current rendering of ``_readme.md.j2``.
4. No number is typed by hand: rendering a template with every placeholder masked must leave no
   digit in its prose. HTML comments and link targets are structure, not prose, and are ignored.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from kashf.ledger.allowlist import public_violations
from kashf.ledger.build import PUBLIC_LEDGER, load_ledger
from kashf.ledger.model import Ledger
from kashf.ledger.render import (
    FINDINGS_DIR,
    README,
    README_TEMPLATE,
    SENTINEL,
    governed_section,
    readme_section,
    render_template,
    rendered_document,
    templates,
)

_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
_LINK_TARGETS = re.compile(r"\]\([^)]*\)")
_DIGIT = re.compile(r"\d")


@dataclass(frozen=True, slots=True, order=True)
class Problem:
    path: str
    line: int
    message: str

    def render(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        return f"{location}: {self.message}"


def hand_typed_numbers(path: str, masked: str) -> list[Problem]:
    """Digits left in prose once every placeholder is masked are numbers typed by hand."""
    problems = []
    without_comments = _COMMENTS.sub(lambda match: "\n" * match.group(0).count("\n"), masked)
    for number, line in enumerate(without_comments.splitlines(), start=1):
        prose = _LINK_TARGETS.sub("]()", line).replace(SENTINEL, "")
        if _DIGIT.search(prose):
            problems.append(
                Problem(path, number, "a number typed by hand; use a ledger placeholder")
            )
    return problems


def _public_ledger(repo_root: Path) -> tuple[Ledger | None, list[Problem]]:
    path = repo_root / PUBLIC_LEDGER
    shown = PUBLIC_LEDGER.as_posix()
    if not path.is_file():
        return None, [
            Problem(shown, 0, "missing: build it with `kashf --private ledger build --public`")
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems = [
        Problem(shown, 0, problem) for problem in public_violations(payload.get("claims", []))
    ]
    return load_ledger(path), problems


def check_repository(repo_root: Path) -> list[Problem]:
    """Every rule above, against the files of ``repo_root``."""
    ledger, problems = _public_ledger(repo_root)
    if ledger is None:
        return problems
    folder = repo_root / FINDINGS_DIR
    names = templates(repo_root)
    for name in names:
        shown = (FINDINGS_DIR / name).as_posix()
        target = folder / name.removesuffix(".j2")
        expected = rendered_document(repo_root, name, ledger)
        if not target.is_file():
            problems.append(Problem(shown, 0, "not rendered: run `kashf ledger render`"))
        elif target.read_text(encoding="utf-8") != expected:
            problems.append(
                Problem(
                    (FINDINGS_DIR / target.name).as_posix(),
                    0,
                    "drifted from its template and the ledger: run `kashf ledger render`",
                )
            )
        problems.extend(
            hand_typed_numbers(shown, render_template(repo_root, name, ledger, masked=True))
        )
    rendered = {name.removesuffix(".j2") for name in names}
    for document in sorted(folder.glob("*.md")):
        if document.name not in rendered:
            problems.append(
                Problem(
                    (FINDINGS_DIR / document.name).as_posix(),
                    0,
                    "findings document without a template",
                )
            )
    readme_template = folder / README_TEMPLATE
    if readme_template.is_file():
        shown = (FINDINGS_DIR / README_TEMPLATE).as_posix()
        section = governed_section((repo_root / README).read_text(encoding="utf-8"))
        if section is None:
            problems.append(
                Problem(README.as_posix(), 0, "missing the ledger:start / ledger:end markers")
            )
        elif section != readme_section(repo_root, ledger).strip():
            problems.append(
                Problem(README.as_posix(), 0, "ledger section drifted: run `kashf ledger render`")
            )
        problems.extend(
            hand_typed_numbers(
                shown, render_template(repo_root, README_TEMPLATE, ledger, masked=True)
            )
        )
    return sorted(problems)
