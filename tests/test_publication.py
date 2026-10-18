"""Publication modes (full and verdict-only), withheld small values, and the data card."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from strata.analysis import markov
from strata.config import Settings
from strata.contracts import data_card
from strata.contracts.vocab import age_bands, seed_rows
from strata.ledger.allowlist import (
    PUBLIC_CLAIM_IDS,
    VERDICT_KEYS,
    VERDICT_ONLY_IDS,
    ledger_violations,
    public_violations,
    verdict_violations,
)
from strata.ledger.build import public_view, verdict_view
from strata.ledger.model import ClaimRecord, Ledger, Unit
from strata.ledger.registry import ClaimError
from strata.ledger.render import Placeholders
from strata.privacy_guard import Scanner, disk_files, scan

if TYPE_CHECKING:
    from strata.ledger.model import PublishedVerdict


def _verdict(synthetic_ledger: Ledger) -> PublishedVerdict:
    return verdict_view(synthetic_ledger.get("silence_by_channel"))


def test_the_final_allowlist() -> None:
    assert {
        "channel_mix",
        "age_digital_divide",
        "weekday_pattern",
        "benford_purchases",
        "spend_concentration",
    } == PUBLIC_CLAIM_IDS
    assert {"silence_by_channel"} == VERDICT_ONLY_IDS


def test_a_verdict_holds_no_value_interval_or_group_size(synthetic_ledger: Ledger) -> None:
    verdict = _verdict(synthetic_ledger)
    payload = verdict.model_dump(mode="json")
    assert set(payload) <= VERDICT_KEYS
    assert verdict.verdict == "reverses"
    assert "activity_tercile" in verdict.reversing_stratifiers
    assert set(verdict.stratifiers) == {"activity_tercile", "age_band", "tenure_band"}
    assert not any(character.isdigit() for character in verdict.mechanism)
    assert verdict_violations([payload]) == []


def test_verdict_rules_catch_numbers_extra_fields_and_unlisted_claims(
    synthetic_ledger: Ledger,
) -> None:
    payload = _verdict(synthetic_ledger).model_dump(mode="json")
    with_number = {**payload, "mechanism": "The gap shrinks by 40 points."}
    assert any("number in its words" in p for p in verdict_violations([with_number]))
    with_values = {**payload, "values": {"silence_rate_branch_only": 0.4}}
    assert any("beyond the verdict" in p for p in verdict_violations([with_values]))
    unlisted = {**payload, "id": "channel_mix"}
    assert any("not allow-listed" in p for p in verdict_violations([unlisted]))
    wrong_mode = {**payload, "publication": "full"}
    assert any("not marked verdict_only" in p for p in verdict_violations([wrong_mode]))
    assert verdict_violations([{"verdict": "holds"}]) == ["verdict #0 has no id"]


def test_a_verdict_only_claim_cannot_be_published_with_its_numbers() -> None:
    problems = public_violations([{"id": "silence_by_channel", "n": 900}])
    assert problems == ["claim 'silence_by_channel' may only be published as a verdict"]


def test_whole_ledger_rules_cover_claims_and_verdicts(synthetic_ledger: Ledger) -> None:
    verdict = _verdict(synthetic_ledger).model_dump(mode="json")
    bad = {**verdict, "mechanism": "n = 120"}
    problems = ledger_violations({"claims": [{"id": "secret", "n": 900}], "verdicts": [bad]})
    assert len(problems) == 2
    assert ledger_violations([]) == ["ledger JSON is not an object"]
    assert ledger_violations({"claims": {}}) == ["ledger JSON has no list of claims or of verdicts"]


def test_the_guard_refuses_a_verdict_carrying_numbers(
    synthetic_ledger: Ledger, git_repo: Path
) -> None:
    payload = _verdict(synthetic_ledger).model_dump(mode="json")
    payload["mechanism"] = "Silence is 12 points higher."
    path = git_repo / "docs" / "ledger" / "claims.public.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"claims": [], "verdicts": [payload]}), encoding="utf-8")
    findings = scan(disk_files([path], git_repo), Scanner.from_environment(environ={}))
    assert any(f.check == "ledger-allowlist" for f in findings)


def test_a_claim_without_a_robustness_result_has_no_verdict(synthetic_ledger: Ledger) -> None:
    with pytest.raises(ClaimError, match="no robustness result"):
        verdict_view(synthetic_ledger.get("channel_mix"))


def test_verdict_placeholders_render_words_and_refuse_numbers(synthetic_ledger: Ledger) -> None:
    public = Ledger(
        source_kind=synthetic_ledger.source_kind,
        data_fingerprint=synthetic_ledger.data_fingerprint,
        computed_at=synthetic_ledger.computed_at,
        claims=[],
        verdicts=[_verdict(synthetic_ledger)],
    )
    places = Placeholders(public)
    assert places.verdict("silence_by_channel") == "reverses"
    assert places.mechanism("silence_by_channel").startswith("Before stratification")
    assert "activity level (tercile of events)" in places.reversed_by("silence_by_channel")
    assert len(places.stratifiers("silence_by_channel")) == 3
    assert "verdict only" in places.verdict_context("silence_by_channel")
    assert places.text("silence_by_channel", "title").startswith("Who goes silent")
    with pytest.raises(KeyError, match="verdict only"):
        places.value("silence_by_channel", "silence_rate_branch_only")
    with pytest.raises(KeyError, match="verdict only"):
        places.n("silence_by_channel")
    with pytest.raises(KeyError, match="verdict only"):
        places.text("silence_by_channel", "method")
    with pytest.raises(KeyError, match="no published verdict"):
        public.verdict("channel_mix")


def test_placeholders_on_full_claims_read_their_robustness(synthetic_ledger: Ledger) -> None:
    places = Placeholders(synthetic_ledger)
    assert places.verdict("silence_by_channel") == "reverses"
    assert places.reversed_by("silence_by_channel")
    assert places.stratifiers("age_digital_divide")


def _record(value_n: dict[str, int]) -> ClaimRecord:
    return ClaimRecord(
        id="spend_concentration",
        title="t",
        question="q",
        method="m",
        values=dict.fromkeys(value_n, 0.5),
        units=dict.fromkeys(value_n, Unit.SHARE),
        ci=dict.fromkeys(value_n, (0.4, 0.6)),
        value_n=value_n,
        n=600,
        meta={"rows_tops": "top 1%:top_1_share|top 10%:top_10_share"},
        public=True,
        data_fingerprint="0" * 64,
        source_kind="real",
        computed_at=datetime(2026, 1, 1, tzinfo=UTC),
        strata_version="test",
    )


def test_values_on_too_few_clients_are_withheld_with_their_rows() -> None:
    shown = public_view(_record({"top_1_share": 6, "top_10_share": 60, "gini": 600}))
    assert set(shown.values) == {"top_10_share", "gini"}
    assert set(shown.ci) == {"top_10_share", "gini"}
    assert shown.meta["rows_tops"] == "top 10%:top_10_share"
    assert shown.meta["withheld_values"] == "top_1_share"
    assert public_violations([shown.model_dump(mode="json")]) == []
    untouched = _record({"top_10_share": 60})
    assert public_view(untouched) == untouched


def test_spend_concentration_sizes_top_shares_by_their_group(synthetic_ledger: Ledger) -> None:
    record = synthetic_ledger.get("spend_concentration")
    assert record.value_n["gini"] == record.n
    assert record.value_n["top_1_share"] == -(-record.n // 100)
    assert record.meta["rows_tops"].count("|") == 2


def test_the_data_card_lists_every_assumption_and_is_current() -> None:
    root = Settings().repo_root
    document = (root / data_card.DATA_CARD).read_text(encoding="utf-8")
    assert data_card.render(document) == document
    rows = data_card.assumptions()
    assert {row.status for row in rows} == {"assumed", "undocumented"}
    assumed_labels = [
        row["label"] for row in seed_rows("codes_branch_operation") if row["semantics"] == "assumed"
    ]
    assert all(label in document for label in assumed_labels)
    holidays = [row["holiday_date"] for row in seed_rows("bank_holidays")]
    assert all(day in document for day in holidays)
    with pytest.raises(ValueError, match="markers"):
        data_card.render("no markers here")


def test_the_data_card_update_rewrites_only_the_table(tmp_path: Path) -> None:
    card = tmp_path / data_card.DATA_CARD
    card.parent.mkdir(parents=True)
    card.write_text(f"# Card\n\n{data_card.START}\nstale\n{data_card.END}\n", encoding="utf-8")
    data_card.update(tmp_path)
    text = card.read_text(encoding="utf-8")
    assert "stale" not in text
    assert text.startswith("# Card")
    assert "| Seed | Field |" in text


CLIENTS = pd.DataFrame(
    {"client_id": ["a", "b", "c", "d", "e", "f"], "age": [22, 40, 44, 45, 70, None]}
)
MONTH = pd.DataFrame(
    {
        "client_id": ["a", "b", "c", "d", "e", "f"],
        "channel_state": ["branch_only"] * 4 + ["mixed", "branch_only"],
    }
)


def test_state_counts_by_age_band_add_up_to_the_month() -> None:
    by_band = markov.counts_by_band(MONTH, CLIENTS, age_bands())
    assert set(by_band) == {*(band.label for band in age_bands()), "missing"}
    total = np.sum(list(by_band.values()), axis=0)
    np.testing.assert_array_equal(total, markov.initial_counts(MONTH, CLIENTS, markov.NO_POLICY))
    assert by_band["missing"][markov.STATES.index("branch_only")] == 1


def test_a_banded_policy_matches_the_age_filter_on_band_boundaries() -> None:
    by_band = markov.counts_by_band(MONTH, CLIENTS, age_bands())
    under_45 = [band.label for band in age_bands() if band.upper <= 45]
    banded = markov.banded_start(by_band, under_45, 0.5)
    direct = markov.initial_counts(MONTH, CLIENTS, markov.younger_than(45, 0.5))
    np.testing.assert_allclose(banded, direct)
    with pytest.raises(ValueError, match="proportion"):
        markov.banded_start(by_band, under_45, 2.0)
