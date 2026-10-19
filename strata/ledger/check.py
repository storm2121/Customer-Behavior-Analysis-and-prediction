"""``strata ledger check``: the documents say exactly what the ledger says, and nothing else.

Four rules, each a CI failure:

1. ``docs/ledger/claims.public.json`` holds only allow-listed claims and verdicts within the
   publication rules.
2. Every ``docs/findings/*.md`` is the current rendering of its template (no drift, no hand edits),
   and every findings document has a template.
3. The README's ledger-governed section is the current rendering of ``_readme.md.j2``, and
   ``docs/CV_NOTES.md`` the current rendering of ``_cv_notes.md.j2``.
4. No number is typed by hand: rendering a template with every placeholder masked must leave no
   digit in its prose. HTML comments and link targets are structure, not prose, and are ignored.
5. Every line of the CV notes ends with its evidence, and the evidence exists: ``claim `id``` names
   a published claim or verdict, ``registry `model.key...``` a key of the public model registry,
   and a link a file of the repository.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2

from strata.ledger.allowlist import ledger_violations
from strata.ledger.build import PUBLIC_LEDGER, load_ledger
from strata.ledger.model import Ledger
from strata.ledger.render import (
    CV_NOTES,
    CV_TEMPLATE,
    FINDINGS_DIR,
    README,
    README_TEMPLATE,
    SENTINEL,
    cv_notes_document,
    governed_section,
    readme_section,
    render_template,
    rendered_document,
    templates,
)

_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
_EVIDENCE = "Evidence:"
_CLAIM = re.compile(r"claim `([^`]+)`")
_REGISTRY = re.compile(r"registry `([^`]+)`")
_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
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
            Problem(shown, 0, "missing: build it with `strata --private ledger build --public`")
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems = [Problem(shown, 0, problem) for problem in ledger_violations(payload)]
    return load_ledger(path), problems


def _model_cards(repo_root: Path) -> list[Problem]:
    """Public model cards are exactly what ``models/registry.json`` renders (if it exists)."""
    from strata.models.cards import card_problems  # noqa: PLC0415 - avoids a cycle at import
    from strata.models.registry import PUBLIC_REGISTRY, load  # noqa: PLC0415

    path = repo_root / PUBLIC_REGISTRY
    if not path.is_file():
        return []
    registry = load(path)
    problems = [Problem(shown, 0, problem) for shown, problem in card_problems(repo_root, registry)]
    if any(entry.source_kind != "synthetic" for entry in registry.models):
        problems.append(
            Problem(PUBLIC_REGISTRY.as_posix(), 0, "holds a model not trained on synthetic data")
        )
    return problems


def _bullets(text: str) -> list[tuple[int, str]]:
    """Top-level bullets of a document, each with its first line and its continuation joined."""
    bullets: list[tuple[int, str]] = []
    current: tuple[int, str] | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("- "):
            if current is not None:
                bullets.append(current)
            current = (number, line[2:])
        elif current is not None and line.startswith("  ") and line.strip():
            current = (current[0], f"{current[1]} {line.strip()}")
        else:
            if current is not None:
                bullets.append(current)
            current = None
    if current is not None:
        bullets.append(current)
    return bullets


def _registry_keys(repo_root: Path) -> dict[str, Any] | None:
    from strata.models.registry import PUBLIC_REGISTRY, load  # noqa: PLC0415 - avoids a cycle

    path = repo_root / PUBLIC_REGISTRY
    if not path.is_file():
        return None
    return {entry.name: entry.model_dump(mode="json") for entry in load(path).models}


def _has_key(tree: object, dotted: str) -> bool:
    for key in dotted.split("."):
        if not isinstance(tree, dict) or key not in tree:
            return False
        tree = tree[key]
    return True


def evidence_problems(repo_root: Path, ledger: Ledger, text: str) -> list[Problem]:
    """Lines of the CV notes without evidence, or whose evidence does not exist."""
    shown = CV_NOTES.as_posix()
    published = set(ledger.ids) | {record.id for record in ledger.verdicts}
    registry = _registry_keys(repo_root)
    problems = []
    for line, bullet in _bullets(text):
        start = bullet.rfind(_EVIDENCE)
        if start < 0:
            problems.append(Problem(shown, line, "a line without evidence"))
            continue
        evidence = bullet[start:]
        claims = _CLAIM.findall(evidence)
        metrics = _REGISTRY.findall(evidence)
        links = _LINK.findall(evidence)
        if not claims and not metrics and not links:
            problems.append(Problem(shown, line, "the evidence names nothing"))
        problems.extend(
            Problem(shown, line, f"evidence names an unpublished claim: {claim}")
            for claim in claims
            if claim not in published
        )
        problems.extend(
            Problem(shown, line, f"evidence names a metric missing from the registry: {metric}")
            for metric in metrics
            if registry is None or not _has_key(registry, metric)
        )
        problems.extend(
            Problem(shown, line, f"evidence links to a missing file: {target}")
            for target in links
            if not (repo_root / CV_NOTES.parent / target.split("#")[0]).exists()
        )
    return problems


def _cv_notes(repo_root: Path, ledger: Ledger) -> list[Problem]:
    template = repo_root / FINDINGS_DIR / CV_TEMPLATE
    if not template.is_file():
        return []
    shown = (FINDINGS_DIR / CV_TEMPLATE).as_posix()
    try:
        expected = cv_notes_document(repo_root, ledger)
        masked = render_template(repo_root, CV_TEMPLATE, ledger, masked=True)
    except (KeyError, jinja2.TemplateError) as error:
        return [Problem(shown, 0, f"cannot be rendered from the ledger: {error}")]
    problems = hand_typed_numbers(shown, masked)
    target = repo_root / CV_NOTES
    if not target.is_file():
        problems.append(Problem(CV_NOTES.as_posix(), 0, "not rendered: run `strata ledger render`"))
    elif target.read_text(encoding="utf-8") != expected:
        problems.append(
            Problem(
                CV_NOTES.as_posix(),
                0,
                "drifted from its template and the ledger: run `strata ledger render`",
            )
        )
    problems.extend(evidence_problems(repo_root, ledger, expected))
    return problems


def check_repository(repo_root: Path) -> list[Problem]:
    """Every rule above, against the files of ``repo_root``."""
    ledger, problems = _public_ledger(repo_root)
    problems.extend(_model_cards(repo_root))
    if ledger is None:
        return problems
    folder = repo_root / FINDINGS_DIR
    names = templates(repo_root)
    for name in names:
        shown = (FINDINGS_DIR / name).as_posix()
        target = folder / name.removesuffix(".j2")
        try:
            expected = rendered_document(repo_root, name, ledger)
            masked = render_template(repo_root, name, ledger, masked=True)
        except (KeyError, jinja2.TemplateError) as error:
            problems.append(Problem(shown, 0, f"cannot be rendered from the ledger: {error}"))
            continue
        if not target.is_file():
            problems.append(Problem(shown, 0, "not rendered: run `strata ledger render`"))
        elif target.read_text(encoding="utf-8") != expected:
            problems.append(
                Problem(
                    (FINDINGS_DIR / target.name).as_posix(),
                    0,
                    "drifted from its template and the ledger: run `strata ledger render`",
                )
            )
        problems.extend(hand_typed_numbers(shown, masked))
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
                Problem(README.as_posix(), 0, "ledger section drifted: run `strata ledger render`")
            )
        problems.extend(
            hand_typed_numbers(
                shown, render_template(repo_root, README_TEMPLATE, ledger, masked=True)
            )
        )
    problems.extend(_cv_notes(repo_root, ledger))
    return sorted(problems)
