"""``strata anomaly run``: rules, detectors, the ranked queue, its evaluation and its storage.

The queue is written to ``<workspace>/anomaly/queue.json`` and its evaluation to
``evaluation.json``; the budgeted part of the queue is stored in the review store
(``strata.review.db``) so that analysts' feedback can be recorded against it and the queue's
precision measured on their labels.

Precision at k is computed against pseudo-labels (clients flagged by rules of at least two
families), for the combined score and for the statistical detectors alone. The second is the
honest one: the combined score contains the rules that define the pseudo-labels, so its precision is
partly circular; the detectors' precision says whether models that never saw the rules agree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from strata.config import Settings, Zone
from strata.ingest.load import load_manifest
from strata.models.anomaly import scores
from strata.models.anomaly.rules import RULES_FILE, Inputs, evaluate, load_rules
from strata.review import db
from strata.warehouse import connect, read_relation

K_VALUES = (10, 25, 50, 100)
DEFAULT_BUDGET = 50


@dataclass(frozen=True)
class AnomalyResult:
    run_id: int
    budget: int
    n_clients: int
    evaluation: dict[str, Any]
    top: pd.DataFrame
    queue_file: Path


def _tables(settings: Settings, zone: Zone) -> dict[str, pd.DataFrame]:
    con = connect(settings.workspace(zone).warehouse, read_only=True)
    try:
        return {
            name: read_relation(con, relation)
            for name, relation in (
                ("events", "intermediate.int_events"),
                ("months", "intermediate.int_client_month"),
                ("clients", "marts.client_360"),
                ("features", "marts.features_anomaly"),
            )
        }
    finally:
        con.close()


def evaluation(queue: scores.Queue, budget: int) -> dict[str, Any]:
    ranked = queue.alerts["client_id"]
    statistical = queue.alerts.sort_values(
        ["statistical_score", "client_id"], ascending=[False, True], kind="stable"
    )["client_id"]
    ks = sorted({*K_VALUES, budget})
    counts: dict[str, int] = {}
    for hit in queue.hits:
        counts[hit.rule] = counts.get(hit.rule, 0) + 1
    return {
        "pseudo_labels": len(queue.pseudo_labels),
        "base_rate": len(queue.pseudo_labels) / len(ranked) if len(ranked) else float("nan"),
        "rule_hits": counts,
        "precision_at_k": {
            str(k): {
                "combined": scores.precision_at(ranked, queue.pseudo_labels, k),
                "statistical_only": scores.precision_at(statistical, queue.pseudo_labels, k),
            }
            for k in ks
        },
    }


def run(
    settings: Settings, zone: Zone, *, budget: int = DEFAULT_BUDGET, store: bool = True
) -> AnomalyResult:
    workspace = settings.workspace(zone)
    folder = workspace.root / "anomaly"
    if workspace.is_private:
        settings.assert_private_output(folder)
    manifest = load_manifest(workspace)
    tables = _tables(settings, zone)
    rules = load_rules()
    hits = evaluate(
        rules, Inputs.from_tables(tables["events"], tables["months"], tables["clients"])
    )
    queue = scores.combine(
        tables["clients"]["client_id"],
        hits=hits,
        rules=rules,
        forest=scores.forest_scores(tables["features"], settings.seed),
        amounts=scores.amount_scores(tables["events"]),
        digits=scores.benford_scores(tables["events"], rules.benford.min_amounts),
    )
    result = evaluation(queue, budget)
    top = queue.top(budget)
    folder.mkdir(parents=True, exist_ok=True)
    queue_file = folder / "queue.json"
    queue_file.write_text(
        json.dumps(json.loads(top.to_json(orient="records")), indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (folder / "evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    run_id = 0
    if store:
        store_engine = db.engine(db.database_url(settings, zone))
        run_id = db.store_run(
            store_engine,
            zone=zone.value,
            source_kind=manifest.source_kind,
            data_fingerprint=manifest.fingerprint,
            rules_digest=hashlib.sha256(RULES_FILE.read_bytes()).hexdigest(),
            n_clients=len(queue.alerts),
            alerts=[{str(k): v for k, v in row.items()} for row in top.to_dict(orient="records")],
            budget=budget,
        )
        store_engine.dispose()
    return AnomalyResult(
        run_id=run_id,
        budget=budget,
        n_clients=len(queue.alerts),
        evaluation=result,
        top=top,
        queue_file=queue_file,
    )
