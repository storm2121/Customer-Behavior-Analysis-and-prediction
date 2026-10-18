"""``strata report``: one HTML page per variant, rendered from the ledger.

- ``private``: every claim of ``claims.private.json``, with its robustness detail, written under
  ``STRATA_PRIVATE_DIR`` only; it is the artefact for interviews and never enters the repository.
- ``public``: the synthetic demo (every claim computed on synthetic data, labelled as such on every
  section) followed by the allow-listed real claims, written under ``.strata/reports``.

The page is self-contained (Plotly is embedded), so a private report opens offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import jinja2
from plotly.offline import get_plotlyjs

from strata import __version__
from strata.config import PrivacyBoundaryError, Settings, Zone
from strata.ledger.build import PUBLIC_LEDGER, load_ledger
from strata.ledger.formatting import format_interval, format_value
from strata.ledger.model import ClaimRecord, Ledger
from strata.report.figures import figures_for

Variant = Literal["public", "private"]
TEMPLATES = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class Section:
    """A ledger as the report shows it."""

    title: str
    banner: str
    ledger: Ledger


def _environment() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES)),
        autoescape=jinja2.select_autoescape(["html", "j2"]),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals.update(
        value=lambda record, key: format_value(record.values[key], record.units[key]),
        interval=lambda record, key: (
            format_interval(record.ci[key], record.units[key]) if key in record.ci else ""
        ),
        figures=figures_for,
        clients=lambda record, key: f"{record.value_n.get(key, record.n):,}",
    )
    return env


def _sections(settings: Settings, variant: Variant) -> list[Section]:
    if variant == "private":
        workspace = settings.workspace(Zone.PRIVATE)
        ledger = load_ledger(workspace.ledger_dir / "claims.private.json")
        if ledger.source_kind != "real":
            msg = "The private report is built from the real extract's ledger."
            raise PrivacyBoundaryError(msg)
        return [
            Section("Every claim, computed on the real extract", "Private: do not share.", ledger)
        ]
    workspace = settings.workspace(Zone.PUBLIC)
    sections = [
        Section(
            "Synthetic demo",
            "Synthetic data from the public generator: the numbers are illustrative and say "
            "nothing about any bank or client.",
            load_ledger(workspace.ledger_dir / "claims.synthetic.json"),
        )
    ]
    public = settings.repo_root / PUBLIC_LEDGER
    if public.is_file():
        sections.append(
            Section(
                "Allow-listed claims on the real extract",
                "Aggregates published under the allowlist; every other real finding is private.",
                load_ledger(public),
            )
        )
    return sections


def report_path(settings: Settings, variant: Variant) -> Path:
    zone = Zone.PRIVATE if variant == "private" else Zone.PUBLIC
    name = "private_report.html" if variant == "private" else "public_report.html"
    path = settings.workspace(zone).reports_dir / name
    if variant == "private":
        settings.assert_private_output(path)
    return path


def build_report(settings: Settings, variant: Variant) -> Path:
    """Render the report of ``variant`` and return where it was written."""
    path = report_path(settings, variant)
    sections = _sections(settings, variant)
    html = (
        _environment()
        .get_template("report.html.j2")
        .render(
            variant=variant,
            sections=sections,
            plotly_js=get_plotlyjs(),
            generated=datetime.now(UTC).replace(microsecond=0).isoformat(),
            version=__version__,
            ordered=_ordered,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8", newline="\n")
    return path


def _ordered(ledger: Ledger) -> list[ClaimRecord]:
    """Headlines first, then the rest by id."""
    return sorted(ledger.claims, key=lambda record: (not record.headline, record.id))
