"""The public generator: invariants for any seed, quirk rates for the default configuration.

Hypothesis explores seeds and sizes for properties that must hold for every draw (schema, keys,
windows, consistency). The quirk rates are statistical, so they are checked once on the default
1,000-client draw with tolerances wide enough to be stable and narrow enough to be meaningful.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kashf.contracts.extracts import validate
from kashf.contracts.specs import EXTRACTS, WINDOW_END, WINDOW_START, Extract
from kashf.contracts.vocab import online_services
from kashf.holidays import is_bank_holiday
from kashf.synthetic.parametric import (
    CARD_END,
    ParametricConfig,
    SyntheticExtract,
    config_digest,
    daily_weights,
    generate,
    window_date,
    window_day,
)

CARD_KEY = [
    "identifiant_client",
    "date_transaction",
    "montant_local",
    "libelle_nature_transaction",
    "type_origine",
]
CARD_SERVICE = next(s.label for s in online_services() if s.zero_amount_expected)

draws = settings(
    max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow], derandomize=True
)


def small(seed: int, n: int) -> SyntheticExtract:
    return generate(ParametricConfig(n_clients=n, seed=seed))


@draws
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(20, 160))
def test_any_draw_satisfies_the_contracts(seed: int, n: int) -> None:
    extract = small(seed, n)
    for spec in EXTRACTS:
        validate(extract.frames[spec.name], spec)
    assert len(extract.frames[Extract.CLIENTS]) == n
    assert extract.frames[Extract.AGGREGATE]["identifier"].notna().sum() == n


@draws
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(20, 160))
def test_keys_join_and_samples_are_disjoint(seed: int, n: int) -> None:
    frames = small(seed, n).frames
    clients = set(frames[Extract.CLIENTS]["identifiant_client"])
    for name in (Extract.CARD, Extract.ONLINE, Extract.BRANCH):
        assert set(frames[name]["identifiant_client"]) <= clients
    aggregate = set(frames[Extract.AGGREGATE]["identifier"].dropna())
    assert not aggregate & clients
    assert all(identifier.startswith("syn_") for identifier in clients | aggregate)


@draws
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(20, 160))
def test_events_stay_inside_their_windows_and_calendars(seed: int, n: int) -> None:
    frames = small(seed, n).frames
    start, end = pd.Timestamp(WINDOW_START), pd.Timestamp(WINDOW_END)
    card = frames[Extract.CARD]["date_transaction"]
    assert card.between(start, pd.Timestamp(CARD_END)).all()
    for name, column in (
        (Extract.ONLINE, "operation_date_time"),
        (Extract.BRANCH, "date_operation"),
    ):
        dates = frames[name][column]
        assert dates.between(start, end).all()
        assert (dates == dates.dt.normalize()).all()
    branch_days = frames[Extract.BRANCH]["date_operation"]
    assert not (branch_days.dt.dayofweek == 5).any()
    assert not branch_days.dt.date.map(is_bank_holiday).any()


@draws
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(20, 160))
def test_amounts_and_zero_rows_follow_the_extract_rules(seed: int, n: int) -> None:
    frames = small(seed, n).frames
    for name, column in (
        (Extract.CARD, "montant_local"),
        (Extract.ONLINE, "montant_transaction"),
        (Extract.BRANCH, "montant_operation"),
    ):
        assert (frames[name][column].dropna() >= 0).all()
    online = frames[Extract.ONLINE]
    is_zero = online["montant_transaction"] == 0
    is_card_service = online["service"] == CARD_SERVICE
    assert (is_zero == is_card_service).all()
    assert (frames[Extract.CARD]["montant_local"].dropna() <= 40_000).all()


@draws
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(20, 160))
def test_the_aggregate_sample_is_internally_consistent(seed: int, n: int) -> None:
    agg = small(seed, n).frames[Extract.AGGREGATE]
    services = [
        "virement_ponctuel",
        "paiement_facture_recharge_telecom_paiement_vignette",
        "virement_permanent",
        "operations_sur_carte",
        "transfert_cash_express",
        "recharge_eshopping",
        "debit_carte",
        "recharge_binatna",
        "image_cheque_emis",
    ]
    assert (agg[services].sum(axis=1) == agg["nombre_operations"]).all()
    assert (agg.loc[agg["client_has_cih_online"] == 2, "nombre_operations"] == 0).all()
    for stem in ("operation", "montant_transaction"):
        low, mid, high = agg[f"minimum_{stem}"], agg[f"mediane_{stem}"], agg[f"maximum_{stem}"]
        assert ((low <= mid + 1e-9) & (mid <= high + 1e-9)).all()
    loans = agg["credit_immobilier"] + agg["credit_consommation"]
    assert (agg["nombre_total_credits"] >= loans).all()
    assert (agg["nombre_mois_derniere_operation"] <= agg["nombre_mois_premiere_operation"]).all()
    assert set(agg["a_email"]) <= {"TRUE", "FALSE"}


@settings(max_examples=5, deadline=None, derandomize=True)
@given(seed=st.integers(0, 2**32 - 1))
def test_output_is_deterministic_for_a_seed(seed: int) -> None:
    first, second = small(seed, 60), small(seed, 60)
    for spec in EXTRACTS:
        pd.testing.assert_frame_equal(first.frames[spec.name], second.frames[spec.name])
    pd.testing.assert_frame_equal(first.truth, second.truth)


def test_different_seeds_give_different_data() -> None:
    a, b = small(1, 80), small(2, 80)
    assert not a.frames[Extract.CARD].equals(b.frames[Extract.CARD])
    assert config_digest(a.config) != config_digest(b.config)


def test_invalid_configurations_are_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        ParametricConfig(n_clients=0)
    bad = ((1.0, 0.0, 0.0, 0.5),) * 6
    with pytest.raises(ValueError, match="sum to one"):
        ParametricConfig(profile_by_age=bad)


def test_window_day_round_trip() -> None:
    assert window_day(WINDOW_START) == 0
    assert window_date(window_day(CARD_END)) == CARD_END


def test_daily_weights_encode_the_calendar() -> None:
    card = daily_weights("card", "withdrawal")
    assert card.mean() == pytest.approx(1.0, rel=0.05)
    branch = daily_weights("branch", "deposit")
    days = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    assert branch[days.dayofweek == 5].max() == 0.0
    deposit = daily_weights("card", "deposit")
    assert deposit[days.month == 12].mean() > 2 * deposit[days.month == 11].mean()


# ------------------------------------------------------------------ default draw: quirk rates


def card_users(frames: dict[Extract, pd.DataFrame], name: Extract) -> int:
    return int(frames[name]["identifiant_client"].nunique())


def test_channel_coverage_is_in_the_documented_ranges(synthetic_extract: SyntheticExtract) -> None:
    frames = synthetic_extract.frames
    assert 650 <= card_users(frames, Extract.CARD) <= 800
    assert 520 <= card_users(frames, Extract.ONLINE) <= 680
    assert 580 <= card_users(frames, Extract.BRANCH) <= 740
    ids = [
        set(frames[name]["identifiant_client"])
        for name in (Extract.CARD, Extract.ONLINE, Extract.BRANCH)
    ]
    active = set.union(*ids)
    assert 25 <= 1000 - len(active) <= 95
    assert 280 <= len(set.intersection(*ids)) <= 420


def test_exact_repeat_rate_matches_the_configuration(synthetic_extract: SyntheticExtract) -> None:
    card = synthetic_extract.frames[Extract.CARD]
    rate = card.duplicated(CARD_KEY).mean()
    assert rate == pytest.approx(synthetic_extract.config.card_repeat_rate, abs=0.01)


def test_card_quirks(synthetic_extract: SyntheticExtract) -> None:
    card = synthetic_extract.frames[Extract.CARD]
    amounts = card["montant_local"]
    assert ((amounts > 0) & (amounts < 1)).sum() > 0
    withdrawals = card[card["libelle_nature_transaction"] == "WITHDRAWAL"]
    assert (withdrawals["montant_local"] % 100 == 0).mean() >= 0.95
    assert withdrawals["montant_local"].value_counts().index[1] == 2000.0
    early = withdrawals[withdrawals["date_transaction"].dt.day <= 5]["montant_local"].sum()
    share = early / withdrawals["montant_local"].sum()
    assert 0.19 <= share <= 0.30
    ecommerce = card[card["libelle_nature_transaction"] == "ACHATS E COMMERCE"]
    purchases = card[card["libelle_nature_transaction"] == "PURCHASES & SERVICES"]
    assert (ecommerce["type_origine"] == "International").mean() == pytest.approx(0.56, abs=0.05)
    assert ecommerce["montant_local"].median() < purchases["montant_local"].median()
    deposits = card[card["libelle_nature_transaction"] == "Deposit"]
    by_month = deposits.groupby(deposits["date_transaction"].dt.to_period("M")).size()
    december = by_month[pd.Period("2023-12")]
    assert december >= 3 * by_month.drop(pd.Period("2023-12")).mean()


def test_branch_quirks(synthetic_extract: SyntheticExtract) -> None:
    branch = synthetic_extract.frames[Extract.BRANCH]
    assert (branch["operation"] == "VERSEMENT SUR COMPTE").mean() >= 0.7
    amounts = branch["montant_operation"].dropna()
    assert amounts.quantile(0.99) >= 20 * amounts.median()
    assert set(branch["devise"]) == {"MAD"}


def test_planted_monitoring_scenarios_exist(synthetic_extract: SyntheticExtract) -> None:
    scenarios = synthetic_extract.truth["scenario"].value_counts()
    assert scenarios.get("structuring", 0) > 0
    assert scenarios.get("pass_through", 0) > 0


def test_dormancy_falls_with_digital_engagement() -> None:
    config = ParametricConfig()
    a, b, floor, ceiling = config.dormancy_digital
    rates = np.array([3.0, 30.0])
    probabilities = np.clip(a - b * np.log(rates), floor, ceiling)
    assert probabilities[0] > 0.6 > probabilities[1]
    assert config.dormancy_branch_only < probabilities[0]


def test_late_starters_and_dormant_clients_bound_the_activity(
    synthetic_extract: SyntheticExtract,
) -> None:
    truth = synthetic_extract.truth
    assert (truth["active_from"] <= truth["active_to"]).all()
    dormant = truth[truth["dormant"]]
    assert (dormant["active_to"] < WINDOW_END).mean() > 0.9
    assert set(truth["profile"]) == {"none", "branch_only", "digital_only", "mixed"}


def test_a_smaller_aggregate_sample_can_be_requested() -> None:
    config = dataclasses.replace(ParametricConfig(n_clients=40, seed=3), n_aggregate_clients=10)
    frames = generate(config).frames
    assert frames[Extract.AGGREGATE]["identifier"].notna().sum() == 10
