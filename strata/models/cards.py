"""Model cards, rendered from registry entries so that no number in them is typed by hand.

A card is generated entirely by code from one entry of a registry (``models/registry.json`` for the
public cards in ``docs/model_cards/``, the private registry for the private ones). ``strata ledger
check`` fails when a public card differs from what its registry entry renders.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from strata.models.registry import ModelEntry, Registry

CARDS_DIR = Path("docs/model_cards")

INTENDED_USE = (
    "Triage and prioritisation by people who review each case: a ranking of clients to look at "
    "first, never an automatic decision about a client."
)
LIMITATIONS = (
    (
        "One year of data: the outcome windows of consecutive cutoffs overlap, so a backtest's "
        "training labels share calendar weeks with its test labels, and seasonal effects cannot "
        "be told apart from trend."
    ),
    "Code meanings marked `assumed` in the seeds (see `docs/data_card.md`) are used as stated.",
    "Synthetic results show that the pipeline works; they are not evidence about clients.",
)


def _defined(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _number(value: object, digits: int = 3) -> str:
    return f"{value:,.{digits}f}" if _defined(value) else "not defined"


def _share(value: object) -> str:
    return f"{100 * value:.1f}%" if _defined(value) else "not defined"  # type: ignore[operator]


def _metric(item: dict[str, Any], digits: int = 3) -> str:
    low, high = item["ci"]
    return f"{_number(item['value'], digits)} ({_number(low, digits)} to {_number(high, digits)})"


def _ratio(item: dict[str, Any], key: str) -> str:
    return (
        f"{_number(item[key], 2)} ({_number(item['ci_low'], 2)} to {_number(item['ci_high'], 2)})"
    )


SILENCE_ORDER = (
    "majority",
    "activity_rule",
    "glm",
    "lightgbm",
    "lightgbm_isotonic",
    "lightgbm_platt",
)
ADOPTION_ORDER = ("glm", "lightgbm")


def _ordered(scores: dict[str, Any], order: tuple[str, ...]) -> list[tuple[str, Any]]:
    """Models in a fixed order (the registry stores them with sorted keys)."""
    known = [(name, scores[name]) for name in order if name in scores]
    return known + [(name, value) for name, value in scores.items() if name not in order]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _provenance(entry: ModelEntry) -> list[str]:
    source = (
        "synthetic data from the public generator (the numbers say nothing about any bank)"
        if entry.source_kind == "synthetic"
        else "the real extract (private)"
    )
    return [
        f"- Version `{entry.version}`, trained {entry.trained_at:%Y-%m-%d}, effort {entry.effort}.",
        f"- Data: {source}; fingerprint `{entry.data_fingerprint[:12]}`.",
        (
            f"- MLflow run `{entry.mlflow_run_id}`."
            if entry.mlflow_run_id
            else "- MLflow run: kept in the local workspace's store, not published."
        ),
    ]


def _silence(entry: ModelEntry) -> list[str]:
    h = entry.headline
    lines = [
        "## What it predicts",
        "",
        (
            "Whether a client with at least one event before a cutoff has no event at all in the "
            "three months after it (silence). Features use events up to the cutoff only; a test "
            "proves that changing anything after the cutoff leaves them unchanged."
        ),
        "",
        "## Validation",
        "",
        (
            f"Trained on the cutoff of {h['train_cutoff']} ({h['n_train']:,} clients, silence "
            f"rate {_share(h['silence_rate_train'])}) and tested on the later cutoff of "
            f"{h['test_cutoff']} ({h['n_test']:,} clients, silence rate "
            f"{_share(h['silence_rate_test'])}). Hyperparameters were tuned on the earliest cutoff "
            "only. Intervals: client bootstrap, 95%."
        ),
        "",
    ]
    rows = [
        [
            name,
            _metric(scores["auc"]),
            _metric(scores["average_precision"]),
            _metric(scores["brier"]),
            _metric(scores["calibration_error"]),
            _metric(scores["lift_at_10"], 2),
        ]
        for name, scores in _ordered(h["metrics"], SILENCE_ORDER)
    ]
    header = ["Model", "ROC AUC", "Average precision", "Brier", "Calibration error", "Lift at 10%"]
    lines += _table(header, rows)
    lines += [
        "",
        (
            "ROC AUC of the trees minus the GLM, paired on the same clients: "
            f"{_metric(h['auc_gain_over_glm'])}."
        ),
        "",
        "Every backtest (the served score: trees with Platt scaling):",
        "",
    ]
    lines += _table(
        ["Train cutoff", "Test cutoff", "ROC AUC", "Lift at 10%"],
        [
            [b["train_cutoff"], b["test_cutoff"], _metric(b["auc"]), _metric(b["lift_at_10"], 2)]
            for b in h["backtests"]
        ],
    )
    lines += ["", "## What drives the score", "", "Mean absolute SHAP value (log-odds):", ""]
    lines += _table(
        ["Feature", "Mean absolute SHAP"],
        [[name, _number(value)] for name, value in h["shap_top"].items()],
    )
    lines += ["", "Odds ratios of the GLM (continuous features per standard deviation):", ""]
    lines += _table(
        ["Feature", "Odds ratio (95% CI)"],
        [[name, _ratio(v, "odds_ratio")] for name, v in h["odds_ratios_top"].items()],
    )
    return lines


def _survival(entry: ModelEntry) -> list[str]:
    h = entry.headline
    lines = [
        "## What it describes",
        "",
        (
            "How long clients go between active days (recurrent gaps, the last one censored at the "
            "end of the window), by channel profile, and a Cox model of the gap with standard "
            "errors clustered by client. Medians carry client-bootstrap intervals."
        ),
        "",
    ]
    rows = [
        [
            g["group"],
            f"{g['n_clients']:,}",
            (
                f"{_number(g['median'], 1)} ({_number(g['median_ci'][0], 1)} to "
                f"{_number(g['median_ci'][1], 1)})"
            ),
            _share(g["survival_30"]),
        ]
        for g in h["gap_by_profile"]
    ]
    header = ["Channel profile", "Clients", "Median gap in days (95% CI)", "Waiting after 30 days"]
    lines += _table(header, rows)
    lines += [
        "",
        f"Log-rank test across profiles: p = {_number(h['logrank_p'], 4)}.",
        "",
        "Cox hazard ratios:",
        "",
    ]
    lines += _table(
        ["Covariate", "Hazard ratio (95% CI)", "Proportional hazards test p"],
        [
            [name, _ratio(v, "hazard_ratio"), _number(h["ph_test_p_values"].get(name), 4)]
            for name, v in h["hazard_ratios"].items()
        ],
    )
    lines += ["", f"Concordance of the Cox model: {_number(h['concordance'])}."]
    return lines


def _adoption(entry: ModelEntry) -> list[str]:
    lines = [
        "## What it predicts",
        "",
        (
            "Online banking adoption on the aggregate sample: access (`has_online`) and active "
            "use (`active_online`). There is no time axis in that file, so metrics are out-of-fold "
            "over five stratified folds, with client-bootstrap intervals. Online operation counts "
            "define the targets and are never features."
        ),
        "",
    ]
    for target, result in entry.headline["targets"].items():
        lines += [
            f"### {target}",
            "",
            f"Clients: {result['n_clients']:,}; rate {_share(result['rate'])}.",
            "",
        ]
        lines += _table(
            ["Model", "ROC AUC", "Average precision", "Brier", "Lift at 10%"],
            [
                [
                    name,
                    _metric(scores["auc"]),
                    _metric(scores["average_precision"]),
                    _metric(scores["brier"]),
                    _metric(scores["lift_at_10"], 2),
                ]
                for name, scores in _ordered(result["metrics"], ADOPTION_ORDER)
            ],
        )
        lines += ["", "Most important features (share of the trees' gain):", ""]
        lines += _table(
            ["Feature", "Share of gain"],
            [[name, _share(value)] for name, value in result["importance_top"].items()],
        )
        lines.append("")
    return lines


def _segments(entry: ModelEntry) -> list[str]:
    h = entry.headline
    stability = h["stability"]
    lines = [
        "## What it finds",
        "",
        (
            "Behavioural segments of active clients: Gaussian mixtures selected by BIC, checked "
            "against KMeans silhouettes, with stability measured by the adjusted Rand index over "
            "bootstrap refits."
        ),
        "",
        (
            f"Segments chosen: {h['k']}. Stability (mean ARI, 95% interval): "
            f"{_number(stability['mean_ari'])} ({_number(stability['ci_low'])} to "
            f"{_number(stability['ci_high'])})."
        ),
        "",
    ]
    lines += _table(
        ["Segment", "Name", "Clients", "Share"],
        [
            [str(s["segment"]), s["name"], f"{s['n_clients']:,}", _share(s["share"])]
            for s in h["segments"]
        ],
    )
    lines += ["", "BIC by number of segments (lower is better):", ""]
    lines += _table(
        ["k", "BIC", "KMeans silhouette"],
        [
            [k, _number(h["bic"][k], 0), _number(h["silhouette"][k])]
            for k in sorted(h["bic"], key=int)
        ],
    )
    return lines


RENDERERS = {
    "silence": _silence,
    "survival": _survival,
    "adoption": _adoption,
    "segments": _segments,
}


def render(entry: ModelEntry) -> str:
    lines = [
        f"<!-- Generated by `strata promote` from models/registry.json ({entry.name}). -->",
        "",
        f"# Model card: {entry.title}",
        "",
        *_provenance(entry),
        "",
        f"Intended use: {INTENDED_USE}",
        "",
        *RENDERERS[entry.name](entry),
        "",
        "## Limitations",
        "",
        *(f"- {text}" for text in LIMITATIONS),
        "",
    ]
    return "\n".join(lines)


def write_cards(registry: Registry, folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for entry in registry.models:
        path = folder / f"{entry.name}.md"
        path.write_text(render(entry), encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def card_problems(repo_root: Path, registry: Registry) -> list[tuple[str, str]]:
    """``(path, problem)`` for public cards missing, out of date, or without a registry entry."""
    folder = repo_root / CARDS_DIR
    expected = {f"{entry.name}.md": render(entry) for entry in registry.models}
    problems = []
    for name, text in expected.items():
        path = folder / name
        shown = (CARDS_DIR / name).as_posix()
        if not path.is_file():
            problems.append((shown, "missing; run `strata promote`"))
        elif path.read_text(encoding="utf-8") != text:
            problems.append((shown, "drifted from models/registry.json; run `strata promote`"))
    extra = sorted(path.name for path in folder.glob("*.md")) if folder.is_dir() else []
    problems.extend(
        ((CARDS_DIR / name).as_posix(), "no model in the registry")
        for name in extra
        if name not in expected
    )
    return problems
