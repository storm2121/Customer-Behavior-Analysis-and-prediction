"""Public parametric generator: five files shaped like the extract, from hand-set distributions.

Nothing here is fitted to real rows. Every parameter is either a property the architecture states in
public (ARCHITECTURE.md section 5.1: coverage of the three channels, the share of exact repeats, the
zero-amount card-service operations, sub-1-MAD transactions, heavy-tailed branch amounts, round
withdrawals, the December deposit wave, salary-day withdrawals, small international e-commerce) or
a round, plausible value chosen by hand. The numbers the generator produces are illustrative by
construction and are always labelled as synthetic.

Design rules:

- **One seed, independent streams.** A :class:`numpy.random.SeedSequence` is spawned into one child
  stream per component, so changing how branch operations are drawn cannot change the card data.
- **Behaviour first, files last.** Each client gets latent traits (age, channel profile, activity
  rates, an activity period) and events are drawn from them; the writer only formats.
- **Mechanisms, not marginals.** The silence-by-channel pattern is produced by its documented
  mechanism (branch-only clients have few events, so an empty quarter happens by chance), plus a
  slightly higher genuine dormancy for digital-only clients, so that the robustness checker has a
  real reversal to find in the public demo.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from kashf.contracts.extracts import COLUMN_TYPES, cast_to_contract
from kashf.contracts.specs import WINDOW_END, WINDOW_START, Extract, spec_for
from kashf.contracts.vocab import age_bands, branch_operations, card_label, online_label
from kashf.holidays import (
    EID_AL_ADHA_2024,
    EID_AL_FITR_2024,
    PRE_EID_AL_ADHA_WEEK_2024,
    is_bank_holiday,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

GENERATOR_VERSION: Final = "1"

AGE_BANDS: Final = tuple(band.label for band in age_bands())
AGE_BAND_EDGES: Final = tuple(band.lower for band in age_bands()[1:])
PROFILES: Final = ("none", "branch_only", "digital_only", "mixed")

CARD_END: Final = date(2024, 9, 29)
"""The card file stops a few days before the other two, as the extract does (ARCHITECTURE §10)."""


@dataclass(frozen=True, slots=True)
class ParametricConfig:
    """Every knob of the generator, with the reason for its default.

    Shares are probabilities, rates are events per client per year, amounts are MAD.
    """

    n_clients: int = 1000
    seed: int = 42
    n_aggregate_clients: int | None = None
    """Size of the second, pre-aggregated sample. Defaults to ``n_clients``."""

    # Channel profile by age band, rows ordered as AGE_BANDS, columns as PROFILES. Older clients
    # are more often branch-only and less often digital-only (ARCHITECTURE §16, item 1).
    profile_by_age: tuple[tuple[float, float, float, float], ...] = (
        (0.06, 0.06, 0.22, 0.66),
        (0.04, 0.07, 0.36, 0.53),
        (0.05, 0.12, 0.30, 0.53),
        (0.05, 0.20, 0.20, 0.55),
        (0.06, 0.28, 0.10, 0.56),
        (0.07, 0.36, 0.04, 0.53),
    )
    p_card_given_digital: float = 0.88
    p_online_given_digital: float = 0.78

    # Annual activity rates: log-normal across clients (median, log-sd). The card median and tail
    # follow ARCHITECTURE §5.1; online and branch are set lower, branch lowest.
    card_rate: tuple[float, float] = (20.0, 1.35)
    online_rate: tuple[float, float] = (16.0, 1.25)
    branch_rate: tuple[float, float] = (4.5, 0.85)
    max_rate: float = 900.0

    # Activity period: some clients start late, some go dormant before the end of the window.
    # Dormancy encodes the mechanism behind the silence-by-channel finding (ARCHITECTURE §8):
    # infrequent branch visits are a stable habit (low, flat dormancy), whereas low digital
    # engagement is a leading sign of attrition (dormancy falls with the digital activity rate,
    # p = clip(a - b * ln(rate))). Branch-only clients are then silent more often overall, simply
    # because they have few events, while at equal activity digital clients go silent more often.
    p_late_start: float = 0.10
    dormancy_branch_only: float = 0.04
    dormancy_digital: tuple[float, float, float, float] = (1.05, 0.30, 0.05, 0.85)
    """``(a, b, floor, ceiling)``: three digital events a year give 0.72; thirty give the floor."""
    mixed_dormancy_factor: float = 0.5

    # Card category participation and base weights (purchase, ecommerce, withdrawal, deposit,
    # refund, transfer_load, transfer_send). About half of card users ever buy online.
    card_category_keys: tuple[str, ...] = (
        "purchase",
        "ecommerce",
        "withdrawal",
        "deposit",
        "refund",
        "transfer_load",
        "transfer_send",
    )
    card_participation: tuple[float, ...] = (0.85, 0.50, 0.80, 0.20, 0.08, 0.03, 0.02)
    card_weights: tuple[float, ...] = (0.50, 0.25, 0.18, 0.04, 0.01, 0.01, 0.01)
    international_share: dict[str, float] = field(
        default_factory=lambda: {
            "purchase": 0.08,
            "ecommerce": 0.56,
            "withdrawal": 0.02,
            "deposit": 0.0,
            "refund": 0.30,
            "transfer_load": 0.30,
            "transfer_send": 0.30,
        }
    )
    december_deposit_multiplier: float = 3.0
    p_december_depositor: float = 0.10
    salary_days_multiplier: float = 1.6
    pre_eid_withdrawal_multiplier: float = 1.3

    # Online services: participation and base weights (bill pay, card service, e-shopping
    # recharge, Cash Express, card debit, Binatna). Bill pay dominates (ARCHITECTURE §5.1).
    online_service_keys: tuple[str, ...] = (
        "bill_pay",
        "card_service",
        "eshopping_recharge",
        "cash_express",
        "card_debit",
        "binatna_recharge",
    )
    online_participation: tuple[float, ...] = (0.95, 0.35, 0.15, 0.08, 0.08, 0.02)
    online_weights: tuple[float, ...] = (0.87, 0.05, 0.04, 0.02, 0.015, 0.005)

    # Branch operations: weights by seed category; cash deposits dominate (ARCHITECTURE §5.1).
    branch_category_weights: dict[str, float] = field(
        default_factory=lambda: {
            "deposit": 0.82,
            "withdrawal": 0.13,
            "transfer": 0.008,
            "card": 0.012,
            "fx": 0.004,
            "payment": 0.006,
            "fee": 0.006,
            "adjustment": 0.004,
        }
    )
    pre_eid_deposit_multiplier: float = 2.2
    p_pre_eid_depositor: float = 0.06
    """Share of branch users who make an extra deposit in the week before Eid al-Adha."""
    branch_regular_share: float = 0.75
    branch_regularity: float = 3.0
    """Gamma shape of the gaps between habitual branch visits (1 would be Poisson; higher is
    more regular). A habit of regular visits is what keeps infrequent branch users from going
    silent by chance (ARCHITECTURE §8)."""

    # Quirks (ARCHITECTURE §5.1).
    card_repeat_rate: float = 0.031
    online_repeat_rate: float = 0.01
    branch_repeat_rate: float = 0.005
    sub_one_mad_rate: float = 0.0026
    psychological_price_rate: float = 0.06
    non_round_withdrawal_rate: float = 0.02
    card_null_amount_rate: float = 0.0001
    card_null_category_rate: float = 0.002
    branch_null_amount_rate: float = 0.0005
    marital_null_rate: float = 0.001
    implausible_children_rate: float = 0.002
    profession_missing_rate: float = 0.47

    # Monitoring scenarios planted for the anomaly layer (M6), as shares of branch users.
    structuring_share: float = 0.015
    pass_through_share: float = 0.01
    structuring_threshold: float = 100_000.0

    def __post_init__(self) -> None:
        if self.n_clients < 1:
            msg = "n_clients must be positive"
            raise ValueError(msg)
        for row in self.profile_by_age:
            if not np.isclose(sum(row), 1.0):
                msg = "each profile_by_age row must sum to one"
                raise ValueError(msg)

    @property
    def aggregate_size(self) -> int:
        return self.n_aggregate_clients if self.n_aggregate_clients is not None else self.n_clients


@dataclass(frozen=True, slots=True)
class SyntheticExtract:
    """The five typed frames plus the latent truth they were drawn from."""

    frames: dict[Extract, pd.DataFrame]
    truth: pd.DataFrame
    config: ParametricConfig


# --------------------------------------------------------------------------- calendar intensities

_DAYS: Final = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
_DOW: Final = _DAYS.dayofweek.to_numpy()
_DOM: Final = _DAYS.day.to_numpy()
_MONTH: Final = _DAYS.month.to_numpy()

# Day-of-week shapes (Monday first). Card spend drifts up towards Saturday; online is quieter on
# Sunday; branches work Monday to Friday, with Monday the busiest and a token weekend trickle of
# postings. Hand-specified.
_CARD_DOW: Final = np.array([0.95, 0.97, 1.0, 1.02, 1.05, 1.08, 0.93])
_ONLINE_DOW: Final = np.array([1.02, 1.04, 1.02, 1.02, 0.99, 0.96, 0.85])
_BRANCH_DOW: Final = np.array([1.15, 1.05, 0.95, 0.95, 1.0, 0.0, 0.005])


def _mask(period_start: date, period_end: date) -> npt.NDArray[np.bool_]:
    days = _DAYS.date
    return np.array([period_start <= day <= period_end for day in days])


_BANK_HOLIDAY: Final = np.array([is_bank_holiday(day) for day in _DAYS.date])
_PRE_EID: Final = _mask(PRE_EID_AL_ADHA_WEEK_2024.start, PRE_EID_AL_ADHA_WEEK_2024.end)
_EIDS: Final = _mask(EID_AL_FITR_2024.start, EID_AL_FITR_2024.end) | _mask(
    EID_AL_ADHA_2024.start, EID_AL_ADHA_2024.end
)
_CARD_COVERAGE: Final = _DAYS.date <= CARD_END


def _normalise(weights: FloatArray, coverage: npt.NDArray[np.bool_]) -> FloatArray:
    """Zero outside coverage, mean one inside, so a weight of 6 means six times a normal day."""
    out = np.where(coverage, weights, 0.0)
    return out / out[coverage].mean()


def daily_weights(
    channel: str, category: str, config: ParametricConfig | None = None
) -> FloatArray:
    """Relative intensity of one channel and category on every day of the window."""
    config = config or ParametricConfig()
    if channel == "card":
        weights = _CARD_DOW[_DOW].astype(np.float64)
        weights = np.where(_EIDS, weights * 0.8, weights)
        if category == "withdrawal":
            weights = np.where(_DOM <= 5, weights * config.salary_days_multiplier, weights)
            weights = np.where(_PRE_EID, weights * config.pre_eid_withdrawal_multiplier, weights)
        if category == "deposit":
            weights = np.where(_MONTH == 12, weights * config.december_deposit_multiplier, weights)
        return _normalise(weights, _CARD_COVERAGE)
    if channel == "online":
        weights = _ONLINE_DOW[_DOW].astype(np.float64)
        return _normalise(weights, np.ones(len(_DAYS), dtype=bool))
    weights = _BRANCH_DOW[_DOW].astype(np.float64)
    weights = np.where(_BANK_HOLIDAY, 0.0, weights)
    if category == "deposit":
        weights = np.where(_PRE_EID, weights * config.pre_eid_deposit_multiplier, weights)
    return _normalise(weights, np.ones(len(_DAYS), dtype=bool))


# --------------------------------------------------------------------------- helpers


def _lognormal(rng: np.random.Generator, median: float, sigma: float, size: int) -> FloatArray:
    return np.asarray(rng.lognormal(np.log(median), sigma, size), dtype=np.float64)


def _identifiers(rng: np.random.Generator, n: int, taken: set[str]) -> list[str]:
    """Opaque synthetic identifiers. The ``syn_`` prefix and the length keep them distinct from
    the real identifier format, so the privacy guard never confuses the two."""
    out: list[str] = []
    while len(out) < n:
        candidate = "syn_" + rng.bytes(6).hex()
        if candidate not in taken:
            taken.add(candidate)
            out.append(candidate)
    return out


def _age_band_index(ages: IntArray) -> IntArray:
    return np.searchsorted(np.array(AGE_BAND_EDGES), ages, side="right").astype(np.int64)


def _choose(rng: np.random.Generator, n: int, weights: npt.ArrayLike) -> IntArray:
    p = np.asarray(weights, dtype=np.float64)
    return rng.choice(len(p), size=n, p=p / p.sum()).astype(np.int64)


# --------------------------------------------------------------------------- clients


def _clients(rng: np.random.Generator, config: ParametricConfig) -> pd.DataFrame:
    n = config.n_clients
    # Age: a young-adult core with a long right tail, clipped to the extract's documented range.
    component = _choose(rng, n, np.array([0.55, 0.30, 0.15]))
    centres = np.array([30.0, 45.0, 64.0])[component]
    spreads = np.array([5.5, 8.0, 9.0])[component]
    age = np.clip(np.rint(rng.normal(centres, spreads)), 20, 99).astype(np.int64)
    # Tenure: most clients are recent, a minority are long-standing; never longer than adulthood.
    long_standing = rng.random(n) < np.clip((age - 25) / 60, 0.05, 0.6)
    scale = np.where(long_standing, 10.0, 1.8)
    tenure = np.clip(2 + np.floor(rng.exponential(scale)), 2, np.minimum(34, age - 16))
    gender = np.where(rng.random(n) < 0.55, 2, 1)
    p_married = np.clip((age - 22) / 30, 0.05, 0.8)
    marital = np.where(rng.random(n) < p_married, 2, 1)
    marital = np.where((age > 40) & (rng.random(n) < 0.08), 3, marital)
    marital = np.where((age > 60) & (rng.random(n) < 0.12), 4, marital)
    children = np.where(marital == 1, rng.poisson(0.15, n), rng.poisson(1.6, n))
    implausible = rng.random(n) < config.implausible_children_rate
    children = np.where(implausible, rng.integers(16, 31, n), children)
    missing_profession = rng.random(n) < config.profession_missing_rate
    # Profession codes: a skewed (Zipf-like) draw over a seeded permutation of the code list.
    sector = _skewed_codes(rng, n, 38)
    profession = _skewed_codes(rng, n, 42)
    category = _skewed_codes(rng, n, 16)
    frame = pd.DataFrame(
        {
            "identifiant_client": _identifiers(rng, n, set()),
            "anciennete_client": tenure.astype(np.int64),
            "age": age,
            "genre": gender.astype(np.int64),
            "situation_familiale": marital.astype(np.int64),
            "nombre_d_enfants": children.astype(np.int64),
            "secteur_professionnel": sector,
            "profession": profession,
            "categorie_socio_professionelle": category,
        }
    )
    frame = _typed(frame, Extract.CLIENTS)
    frame.loc[rng.random(n) < config.marital_null_rate, "situation_familiale"] = pd.NA
    for column in ("secteur_professionnel", "profession", "categorie_socio_professionelle"):
        frame.loc[missing_profession, column] = pd.NA
    return frame


def _skewed_codes(rng: np.random.Generator, n: int, size: int) -> IntArray:
    order = rng.permutation(np.arange(1, size + 1))
    weights = 1.0 / np.arange(1, size + 1) ** 1.1
    return order[_choose(rng, n, weights)].astype(np.int64)


# --------------------------------------------------------------------------- latent behaviour


@dataclass(frozen=True, slots=True)
class _Latent:
    profile: npt.NDArray[np.str_]
    has_card: npt.NDArray[np.bool_]
    has_online: npt.NDArray[np.bool_]
    has_branch: npt.NDArray[np.bool_]
    card_rate: FloatArray
    online_rate: FloatArray
    branch_rate: FloatArray
    start: IntArray
    end: IntArray
    dormant: npt.NDArray[np.bool_]


def _latent(rng: np.random.Generator, clients: pd.DataFrame, config: ParametricConfig) -> _Latent:
    n = len(clients)
    band = _age_band_index(clients["age"].to_numpy(dtype=np.int64))
    table = np.array(config.profile_by_age)
    draws = rng.random(n)
    cumulative = np.cumsum(table[band], axis=1)
    profile_index = (draws[:, None] > cumulative).sum(axis=1)
    profile = np.array(PROFILES)[profile_index]
    digital = np.isin(profile, ["digital_only", "mixed"])
    has_branch = np.isin(profile, ["branch_only", "mixed"])
    has_card = digital & (rng.random(n) < config.p_card_given_digital)
    has_online = digital & (rng.random(n) < config.p_online_given_digital)
    has_card = has_card | (digital & ~has_card & ~has_online)

    def rates(params: tuple[float, float]) -> FloatArray:
        return np.minimum(_lognormal(rng, params[0], params[1], n), config.max_rate)

    card_rate = rates(config.card_rate)
    online_rate = rates(config.online_rate)
    branch_rate = rates(config.branch_rate)
    n_days = len(_DAYS)
    late = rng.random(n) < config.p_late_start
    start = np.where(late, rng.integers(30, 240, n), 0).astype(np.int64)
    digital_rate = card_rate * has_card + online_rate * has_online
    a, b, floor, ceiling = config.dormancy_digital
    digital_dormancy = np.clip(a - b * np.log(np.maximum(digital_rate, 1e-9)), floor, ceiling)
    p_dormant = np.select(
        [profile == "branch_only", profile == "digital_only", profile == "mixed"],
        [
            np.full(n, config.dormancy_branch_only),
            digital_dormancy,
            digital_dormancy * config.mixed_dormancy_factor,
        ],
        default=0.0,
    )
    dormant = rng.random(n) < p_dormant
    stop = start + 45 + (rng.random(n) * (n_days - 1 - start - 45)).astype(np.int64)
    end = np.where(dormant, np.minimum(stop, n_days - 1), n_days - 1).astype(np.int64)
    return _Latent(
        profile=profile,
        has_card=has_card,
        has_online=has_online,
        has_branch=has_branch,
        card_rate=card_rate,
        online_rate=online_rate,
        branch_rate=branch_rate,
        start=start,
        end=end,
        dormant=dormant,
    )


def _draw_days(
    rng: np.random.Generator, rate: float, weights: FloatArray, start: int, end: int
) -> IntArray:
    """Event days for one client and one stream: Poisson count, then days by intensity."""
    window = weights[start : end + 1]
    expected = rate / 365.25 * float(window.sum())
    count = int(rng.poisson(expected))
    if count == 0 or window.sum() <= 0:
        return np.empty(0, dtype=np.int64)
    return (start + rng.choice(len(window), size=count, p=window / window.sum())).astype(np.int64)


def _renewal_days(
    rng: np.random.Generator,
    *,
    rate: float,
    shape: float,
    start: int,
    end: int,
    open_days: IntArray,
) -> IntArray:
    """Regular visits: gamma-distributed gaps and a random phase, moved to the next open day."""
    if rate <= 0 or end < start:
        return np.empty(0, dtype=np.int64)
    scale = 365.25 / rate / shape
    times: list[float] = []
    current = start + rng.uniform() * float(rng.gamma(shape, scale))
    while current <= end:
        times.append(current)
        current += float(rng.gamma(shape, scale))
    if not times:
        return np.empty(0, dtype=np.int64)
    positions = np.searchsorted(open_days, np.floor(np.array(times)).astype(np.int64))
    snapped = open_days[positions[positions < len(open_days)]]
    return snapped[snapped <= end].astype(np.int64)


def _participation(
    rng: np.random.Generator, n: int, participation: tuple[float, ...], weights: tuple[float, ...]
) -> FloatArray:
    """Per-client stream weights: who ever uses each category, renormalised."""
    mask = rng.random((n, len(participation))) < np.array(participation)
    empty = ~mask.any(axis=1)
    mask[empty, 0] = True
    raw = mask * np.array(weights)
    return np.asarray(raw / raw.sum(axis=1, keepdims=True), dtype=np.float64)


# --------------------------------------------------------------------------- card


def _card_amounts(
    rng: np.random.Generator, key: str, n: int, config: ParametricConfig
) -> FloatArray:
    if key == "purchase":
        amounts = _lognormal(rng, 120.0, 0.95, n)
        psych = rng.random(n) < config.psychological_price_rate
        # Prices that lead with a nine (9.90, 99, 990, ...): the source of the digit-9 excess.
        nines = np.array([9.5, 9.9, 9.99, 95.0, 99.0, 99.9, 950.0, 990.0, 999.0])
        amounts = np.where(psych, rng.choice(nines, n), amounts)
        tiny = rng.random(n) < config.sub_one_mad_rate
        amounts = np.where(tiny, rng.integers(1, 100, n) / 100.0, amounts)
    elif key == "ecommerce":
        amounts = _lognormal(rng, 75.0, 1.15, n)
    elif key == "withdrawal":
        notes = np.array([200, 2000, 500, 1000, 100, 300, 400, 1500, 3000, 2500, 4000, 5000])
        weights = np.array(
            [0.19, 0.17, 0.16, 0.15, 0.10, 0.07, 0.05, 0.03, 0.03, 0.02, 0.015, 0.015]
        )
        amounts = notes[_choose(rng, n, weights)].astype(np.float64)
        odd = rng.random(n) < config.non_round_withdrawal_rate
        amounts = np.where(odd, np.round(_lognormal(rng, 300.0, 0.8, n), 2), amounts)
    elif key == "deposit":
        amounts = np.round(_lognormal(rng, 400.0, 1.3, n), -1)
        amounts = np.maximum(amounts, 20.0)
    elif key == "refund":
        amounts = _lognormal(rng, 85.0, 1.2, n)
    elif key == "transfer_load":
        amounts = _lognormal(rng, 700.0, 1.0, n)
    else:
        amounts = _lognormal(rng, 200.0, 0.8, n)
    return np.clip(np.round(amounts, 2), 0.01, 40_000.0)


def _stack(parts: list[IntArray]) -> IntArray:
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _card(
    rng: np.random.Generator, clients: pd.DataFrame, latent: _Latent, config: ParametricConfig
) -> pd.DataFrame:
    keys = config.card_category_keys
    n = len(clients)
    weights = _participation(rng, n, config.card_participation, config.card_weights)
    december_depositor = rng.random(n) < config.p_december_depositor
    day_weights = {key: daily_weights("card", key, config) for key in keys}
    december = np.flatnonzero((_MONTH == 12) & _CARD_COVERAGE)
    deposit_key = keys.index("deposit")
    ids = clients["identifiant_client"].to_numpy()
    rows_client: list[IntArray] = []
    rows_day: list[IntArray] = []
    rows_key: list[IntArray] = []

    def emit(client: int, days: IntArray, key_index: int) -> None:
        if days.size:
            rows_client.append(np.full(days.size, client, dtype=np.int64))
            rows_day.append(days)
            rows_key.append(np.full(days.size, key_index, dtype=np.int64))

    for i in np.flatnonzero(latent.has_card):
        start, end = int(latent.start[i]), int(latent.end[i])
        for k, key in enumerate(keys):
            if weights[i, k] > 0:
                rate = float(latent.card_rate[i] * weights[i, k])
                emit(int(i), _draw_days(rng, rate, day_weights[key], start, end), k)
        if december_depositor[i]:
            # Breadth of the December wave: clients who rarely deposit make one or two December
            # deposits, on top of the deeper December activity of regular depositors.
            open_days = december[(december >= start) & (december <= end)]
            if open_days.size:
                extra = rng.choice(open_days, size=1 + int(rng.poisson(0.6)), replace=True)
                emit(int(i), extra.astype(np.int64), deposit_key)

    client_index, day_index, key_index = _stack(rows_client), _stack(rows_day), _stack(rows_key)
    amounts = np.zeros(len(client_index), dtype=np.float64)
    origin = np.full(len(client_index), "National", dtype=object)
    for k, key in enumerate(keys):
        selected = key_index == k
        count = int(selected.sum())
        amounts[selected] = _card_amounts(rng, key, count, config)
        international = rng.random(count) < config.international_share[key]
        origin[selected] = np.where(international, "International", "National")
    labels = np.array([card_label(key) for key in keys], dtype=object)[key_index]
    frame = pd.DataFrame(
        {
            "identifiant_client": ids[client_index],
            "date_transaction": _DAYS[day_index],
            "montant_local": amounts,
            "libelle_nature_transaction": labels,
            "type_origine": origin,
        }
    )
    frame = _typed(_with_repeats(rng, frame, config.card_repeat_rate), Extract.CARD)
    size = len(frame)
    frame.loc[rng.random(size) < config.card_null_amount_rate, "montant_local"] = np.nan
    null_category = rng.random(size) < config.card_null_category_rate
    frame.loc[null_category, "libelle_nature_transaction"] = pd.NA
    return frame


# --------------------------------------------------------------------------- online


def _online_amounts(rng: np.random.Generator, key: str, n: int) -> FloatArray:
    if key == "bill_pay":
        # Mobile top-ups come in round denominations; bills do not.
        top_up = rng.random(n) < 0.55
        denominations = np.array([10.0, 20.0, 30.0, 50.0, 100.0, 200.0])
        round_amounts = denominations[
            _choose(rng, n, np.array([0.25, 0.30, 0.10, 0.15, 0.15, 0.05]))
        ]
        bills = np.round(_lognormal(rng, 150.0, 1.0, n), 2)
        amounts = np.where(top_up, round_amounts, bills)
    elif key == "card_service":
        amounts = np.zeros(n)
    elif key == "eshopping_recharge":
        amounts = np.round(_lognormal(rng, 150.0, 0.9, n), 0)
    elif key == "cash_express":
        amounts = np.round(_lognormal(rng, 300.0, 0.9, n), -1)
    elif key == "card_debit":
        amounts = np.round(_lognormal(rng, 100.0, 1.2, n), 2)
    else:
        amounts = np.round(_lognormal(rng, 800.0, 0.7, n), -2)
    floor = 0.0 if key == "card_service" else 1.0
    return np.clip(amounts, floor, 48_000.0)


def _online(
    rng: np.random.Generator, clients: pd.DataFrame, latent: _Latent, config: ParametricConfig
) -> pd.DataFrame:
    keys = config.online_service_keys
    n = len(clients)
    weights = _participation(rng, n, config.online_participation, config.online_weights)
    day_weights = daily_weights("online", "all", config)
    ids = clients["identifiant_client"].to_numpy()
    rows_client: list[IntArray] = []
    rows_day: list[IntArray] = []
    for i in np.flatnonzero(latent.has_online):
        rate = float(latent.online_rate[i])
        days = _draw_days(rng, rate, day_weights, int(latent.start[i]), int(latent.end[i]))
        if days.size:
            rows_client.append(np.full(days.size, i, dtype=np.int64))
            rows_day.append(days)
    client_index, day_index = _stack(rows_client), _stack(rows_day)
    # Service of each operation by inverse CDF on the client's own service weights.
    cumulative = np.cumsum(weights[client_index], axis=1)
    key_index = (rng.random(len(client_index))[:, None] > cumulative).sum(axis=1)
    key_index = np.minimum(key_index, len(keys) - 1).astype(np.int64)
    amounts = np.zeros(len(client_index), dtype=np.float64)
    for k, key in enumerate(keys):
        selected = key_index == k
        amounts[selected] = _online_amounts(rng, key, int(selected.sum()))
    labels = np.array([online_label(key) for key in keys], dtype=object)
    frame = pd.DataFrame(
        {
            "identifiant_client": ids[client_index],
            "service": labels[key_index],
            "operation_date_time": _DAYS[day_index],
            "montant_transaction": amounts,
        }
    )
    return _typed(_with_repeats(rng, frame, config.online_repeat_rate), Extract.ONLINE)


# --------------------------------------------------------------------------- branch


def _branch_amounts(rng: np.random.Generator, category: str, n: int) -> FloatArray:
    if category == "deposit":
        # A heavy right tail: everyday deposits plus occasional large ones (ARCHITECTURE §5.1).
        large = rng.random(n) < 0.08
        amounts = np.where(
            large,
            np.round(_lognormal(rng, 30_000.0, 1.0, n), -2),
            np.round(_lognormal(rng, 1_500.0, 1.1, n), -1),
        )
    elif category == "withdrawal":
        amounts = np.round(_lognormal(rng, 3_500.0, 1.2, n), -1)
    elif category == "transfer":
        amounts = np.round(_lognormal(rng, 10_000.0, 1.0, n), 2)
    elif category == "fee":
        amounts = rng.choice(np.array([20.0, 50.0, 100.0, 150.0, 200.0]), n)
    else:
        amounts = np.round(_lognormal(rng, 500.0, 1.0, n), 2)
    return np.clip(amounts, 3.0, 600_000.0)


BranchRecord = tuple[int, int, str, float, str]
"""(client index, day index, label, amount, direction) of one branch operation."""


def _plant_scenarios(
    rng: np.random.Generator,
    *,
    latent: _Latent,
    config: ParametricConfig,
    open_days: IntArray,
    records: list[BranchRecord],
    scenario: npt.NDArray[np.object_],
) -> None:
    """Add the monitoring patterns the anomaly layer must find, and label who carries them."""
    branch_users = np.flatnonzero(latent.has_branch)
    structuring = rng.random(len(branch_users)) < config.structuring_share
    for i in branch_users[structuring]:
        # Several deposits, each below the reporting threshold, summing above it within 30 days.
        scenario[i] = "structuring"
        first = int(rng.choice(open_days[open_days < len(_DAYS) - 35]))
        count = int(rng.integers(3, 6))
        parts = (
            rng.dirichlet(np.ones(count)) * config.structuring_threshold * rng.uniform(1.05, 1.6)
        )
        parts = np.minimum(parts, config.structuring_threshold * 0.95)
        candidates = open_days[(open_days >= first) & (open_days < first + 30)]
        days = np.sort(rng.choice(candidates, size=count, replace=True))
        for day, amount in zip(days, parts, strict=True):
            records.append(
                (int(i), int(day), "VERSEMENT SUR COMPTE", float(np.round(amount, -2)), "C")
            )
    pass_through = (rng.random(len(branch_users)) < config.pass_through_share) & ~structuring
    for i in branch_users[pass_through]:
        # A large deposit withdrawn again within a week.
        scenario[i] = "pass_through"
        day = int(rng.choice(open_days[open_days < len(_DAYS) - 10]))
        amount = float(np.round(rng.uniform(40_000, 150_000), -3))
        records.append((int(i), day, "VERSEMENT SUR COMPTE", amount, "C"))
        following = open_days[(open_days > day) & (open_days <= day + 7)]
        later = int(rng.choice(following)) if following.size else day
        records.append(
            (
                int(i),
                later,
                "RETRAIT BORDEREAU",
                float(np.round(amount * rng.uniform(0.8, 0.98), -2)),
                "D",
            )
        )


def _branch(
    rng: np.random.Generator, clients: pd.DataFrame, latent: _Latent, config: ParametricConfig
) -> tuple[pd.DataFrame, npt.NDArray[np.str_]]:
    operations = branch_operations()
    categories = list(config.branch_category_weights)
    category_weights = np.array([config.branch_category_weights[c] for c in categories])
    deposit_weights = daily_weights("branch", "deposit", config)
    other_weights = daily_weights("branch", "other", config)
    ids = clients["identifiant_client"].to_numpy()
    scenario = np.full(len(clients), "none", dtype=object)
    records: list[BranchRecord] = []

    def label_for(category: str) -> tuple[str, str]:
        options = [op for op in operations if op.category == category]
        # Within a category the first label is the everyday one; the rest are rare variants.
        weights = np.array([8.0] + [1.0] * (len(options) - 1))
        chosen = options[int(_choose(rng, 1, weights)[0])]
        return chosen.label, chosen.assumed_direction

    open_days = np.flatnonzero((_BRANCH_DOW[_DOW] >= 0.5) & ~_BANK_HOLIDAY)
    pre_eid_days = np.flatnonzero(_PRE_EID & (other_weights > 0))
    category_p = category_weights / category_weights.sum()

    def emit(client: int, day: int, category: str) -> None:
        label, direction = label_for(category)
        amount = float(_branch_amounts(rng, category, 1)[0])
        records.append((client, day, label, amount, direction))

    for i in np.flatnonzero(latent.has_branch):
        start, end = int(latent.start[i]), int(latent.end[i])
        rate = float(latent.branch_rate[i])
        # Habitual visits are regular; the rest arrive at random on the branch calendar.
        regular = _renewal_days(
            rng,
            rate=rate * config.branch_regular_share,
            shape=config.branch_regularity,
            start=start,
            end=end,
            open_days=open_days,
        )
        for day in regular:
            emit(int(i), int(day), categories[int(_choose(rng, 1, category_p)[0])])
        occasional_rate = rate * (1 - config.branch_regular_share)
        deposit_rate = occasional_rate * category_p[0]
        for day in _draw_days(rng, deposit_rate, deposit_weights, start, end):
            emit(int(i), int(day), "deposit")
        other_p = category_p[1:] / category_p[1:].sum()
        other_rate = occasional_rate * (1 - category_p[0])
        for day in _draw_days(rng, other_rate, other_weights, start, end):
            emit(int(i), int(day), categories[1 + int(_choose(rng, 1, other_p)[0])])
        if rng.random() < config.p_pre_eid_depositor:
            window = pre_eid_days[(pre_eid_days >= start) & (pre_eid_days <= end)]
            if window.size:
                emit(int(i), int(rng.choice(window)), "deposit")

    _plant_scenarios(
        rng, latent=latent, config=config, open_days=open_days, records=records, scenario=scenario
    )

    frame = pd.DataFrame(
        {
            "identifiant_client": [ids[r[0]] for r in records],
            "date_operation": _DAYS[np.array([r[1] for r in records], dtype=np.int64)],
            "montant_operation": [r[3] for r in records],
            "sens_operation": [r[4] for r in records],
            "operation": [r[2] for r in records],
            "devise": "MAD",
        }
    )
    frame = _typed(_with_repeats(rng, frame, config.branch_repeat_rate), Extract.BRANCH)
    nulls = rng.random(len(frame)) < config.branch_null_amount_rate
    frame.loc[nulls, "montant_operation"] = np.nan
    frame.loc[nulls, "sens_operation"] = pd.NA
    return frame, scenario.astype(str)


# --------------------------------------------------------------------------- aggregate sample


def _aggregate(rng: np.random.Generator, config: ParametricConfig, taken: set[str]) -> pd.DataFrame:
    """The second sample: one pre-aggregated row per client, internally consistent.

    The meaning of several columns is undocumented; the generator makes them consistent with each
    other (counts add up, minima never exceed maxima, clients without online access have no online
    operations) rather than imitating any particular distribution.
    """
    n = config.aggregate_size
    age = np.clip(np.rint(rng.normal(37, 13, n)), 18, 102).astype(np.int64)
    email = rng.random(n) < np.clip(1.15 - age / 75, 0.15, 0.97)
    logit = 1.2 - 0.045 * (age - 35) + 1.6 * email.astype(float)
    has_online = rng.random(n) < 1 / (1 + np.exp(-logit))
    active_contract = rng.random(n) < 0.9

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
    service_rates = np.array([3.0, 8.0, 0.6, 1.2, 0.5, 0.4, 0.2, 0.05, 0.0])
    intensity = _lognormal(rng, 1.0, 1.1, n)
    counts = {
        name: np.where(has_online, rng.poisson(rate * intensity), 0).astype(np.int64)
        for name, rate in zip(services, service_rates, strict=True)
    }
    total_ops = np.sum(np.column_stack(list(counts.values())), axis=1)
    op_stats = _amount_stats(rng, total_ops, median=600.0, sigma=1.2)

    card_types = [
        "carte_cih_utilisee_dans_un_gab_cih",
        "carte_cih_utilisee_chez_un_commerçant_national",
        "carte_cih_utilisee_chez_un_commerçant_international",
    ]
    card_rates = np.array([4.0, 10.0, 3.0])
    has_card = rng.random(n) < 0.75
    card_counts = {
        name: np.where(has_card, rng.poisson(rate * intensity), 0).astype(np.int64)
        for name, rate in zip(card_types, card_rates, strict=True)
    }
    txn_types = {
        "withdrawal": card_counts["carte_cih_utilisee_dans_un_gab_cih"],
        "mastercardmoney_send": np.where(has_card, rng.poisson(0.05, n), 0),
        "achats_e_commerce": np.where(has_card, rng.poisson(0.3 * intensity), 0),
        "purchases_and_services": card_counts["carte_cih_utilisee_chez_un_commerçant_national"],
    }
    total_txn = np.sum(np.column_stack([np.asarray(v) for v in txn_types.values()]), axis=1)
    txn_stats = _amount_stats(rng, total_txn, median=150.0, sigma=1.1)
    first_op = np.where(total_ops > 0, rng.integers(1, 62, n), 0)
    last_op = np.where(total_ops > 0, np.minimum(first_op, rng.integers(0, 12, n)), 0)
    first_txn = np.where(total_txn > 0, rng.integers(1, 46, n), 0)
    last_txn = np.where(total_txn > 0, np.minimum(first_txn, rng.integers(0, 8, n)), 0)
    mortgage = rng.binomial(1, 0.06, n) + rng.binomial(1, 0.01, n)
    consumer = rng.poisson(0.3, n)
    other_credit = rng.poisson(0.05, n)
    international = card_counts["carte_cih_utilisee_chez_un_commerçant_international"] > 0

    zeros = np.zeros(n, dtype=np.int64)
    columns: dict[str, object] = {
        "identifier": _identifiers(rng, n, taken),
        "duree_creation_client_en_annee": np.clip(np.floor(rng.exponential(4.0, n)), 0, 34).astype(
            np.int64
        ),
        "age": age,
        "civilite": rng.integers(0, 2, n),
        "pays_de_naissance": np.where(rng.random(n) < 0.92, 100, rng.integers(1, 190, n)),
        "code_pays_de_residence": np.where(rng.random(n) < 0.95, 85, rng.integers(10, 139, n)),
        "situation_familiale": rng.choice([1, 2, 3, 4], n, p=[0.55, 0.38, 0.04, 0.03]),
        "nombre_d_enfants": np.minimum(rng.poisson(0.9, n), 8),
        "situation_professionelle": rng.integers(0, 19, n),
        "lieu_creation": rng.integers(0, 2, n),
        "a_email": np.where(email, "TRUE", "FALSE"),
        "resident_status": rng.choice([1, 2, 3, 4], n, p=[0.9, 0.05, 0.03, 0.02]),
        "flag_profession": rng.integers(1, 7, n),
        "client_has_cih_online": np.where(has_online, 1, 2),
        "flag_contrat_actif": np.where(active_contract, 1, 2),
        **counts,
        "nombre_operations": total_ops,
        **{
            f"{stat}_operation": op_stats[stat]
            for stat in ("moyenne", "ecart_type", "mediane", "minimum", "maximum")
        },
        "nombre_mois_premiere_operation": first_op,
        "nombre_mois_derniere_operation": last_op,
        "nombre_total_credits": mortgage + consumer + other_credit,
        "credit_immobilier": mortgage,
        "credit_consommation": consumer,
        "flag_debit_credit": rng.choice([-1, 0, 1, 2], n, p=[0.1, 0.6, 0.2, 0.1]),
        **card_counts,
        "carte_nationale_hors_carte_cih_utilisee_dans_un_gab_cih": zeros,
        "carte_internationale_hors_carte_cih_utilisee_dans_un_gab_cih": zeros,
        "deposit": zeros,
        "withdrawal": txn_types["withdrawal"],
        "mastercardmoney_send": txn_types["mastercardmoney_send"],
        "refund": zeros,
        "bill_payment": zeros,
        "achat_sur_gab": zeros,
        "visa_money_transfert_load": zeros,
        "achats_e_commerce": txn_types["achats_e_commerce"],
        "purchases_and_services": txn_types["purchases_and_services"],
        "paiement_vignette": zeros,
        "nombre_transaction": total_txn,
        **{
            f"{stat}_montant_transaction": txn_stats[stat]
            for stat in ("moyenne", "ecart_type", "mediane", "minimum", "maximum")
        },
        "transaction_international": international.astype(np.int64),
        "nombre_mois_premiere_transaction": first_txn,
        "nombre_mois_derniere_transaction": last_txn,
    }
    frame = pd.DataFrame({name: columns[name] for name in spec_for(Extract.AGGREGATE).columns})
    return _typed(frame, Extract.AGGREGATE)


def _amount_stats(
    rng: np.random.Generator, counts: IntArray, *, median: float, sigma: float
) -> dict[str, FloatArray]:
    """Mean, standard deviation, median, minimum and maximum of each client's amounts."""
    stats = {
        key: np.zeros(len(counts))
        for key in ("moyenne", "ecart_type", "mediane", "minimum", "maximum")
    }
    for i, count in enumerate(counts):
        if count <= 0:
            continue
        amounts = np.round(_lognormal(rng, median, sigma, int(min(count, 2000))), 2)
        stats["moyenne"][i] = round(float(amounts.mean()), 8)
        stats["ecart_type"][i] = round(float(amounts.std(ddof=1)) if count > 1 else 0.0, 8)
        stats["mediane"][i] = float(np.median(amounts))
        stats["minimum"][i] = float(amounts.min())
        stats["maximum"][i] = float(amounts.max())
    return stats


# --------------------------------------------------------------------------- shared steps


def _with_repeats(rng: np.random.Generator, frame: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Append exact copies of a random subset of rows (double postings that cannot be told apart).

    ``rate`` is the share of repeat rows in the final file: ``k / (n + k) = rate``.
    """
    n = len(frame)
    if n == 0 or rate <= 0:
        return frame
    k = int(np.round(rate * n / (1 - rate)))
    copies = frame.iloc[rng.choice(n, size=k, replace=True)]
    combined = pd.concat([frame, copies], ignore_index=True)
    date_column = next(
        column for column in combined.columns if column.startswith(("date", "operation_date"))
    )
    order = np.lexsort(
        (combined["identifiant_client"].to_numpy(dtype=str), combined[date_column].to_numpy())
    )
    return combined.iloc[order].reset_index(drop=True)


def _typed(frame: pd.DataFrame, extract: Extract) -> pd.DataFrame:
    """Cast to the contract's pandas dtypes, in the contract's column order."""
    types = COLUMN_TYPES[extract]
    return pd.DataFrame(
        {
            column: cast_to_contract(frame[column], types[column])
            for column in spec_for(extract).columns
        }
    )


def generate(config: ParametricConfig | None = None) -> SyntheticExtract:
    """Draw the five extracts for ``config`` (deterministic for a given seed)."""
    config = config or ParametricConfig()
    streams = np.random.SeedSequence(config.seed).spawn(6)
    rngs = [np.random.default_rng(stream) for stream in streams]
    clients = _clients(rngs[0], config)
    latent = _latent(rngs[1], clients, config)
    card = _card(rngs[2], clients, latent, config)
    online = _online(rngs[3], clients, latent, config)
    branch, scenario = _branch(rngs[4], clients, latent, config)
    aggregate = _aggregate(rngs[5], config, set(clients["identifiant_client"]))
    truth = pd.DataFrame(
        {
            "identifiant_client": clients["identifiant_client"],
            "profile": latent.profile,
            "has_card": latent.has_card,
            "has_online": latent.has_online,
            "has_branch": latent.has_branch,
            "dormant": latent.dormant,
            "active_from": _DAYS[latent.start].date,
            "active_to": _DAYS[latent.end].date,
            "scenario": scenario,
        }
    )
    frames = {
        Extract.CLIENTS: clients,
        Extract.CARD: card,
        Extract.ONLINE: online,
        Extract.BRANCH: branch,
        Extract.AGGREGATE: aggregate,
    }
    return SyntheticExtract(frames=frames, truth=truth, config=config)


def config_digest(config: ParametricConfig) -> str:
    """A short hash of the configuration, recorded in the synthetic marker."""
    return hashlib.sha256(repr(config).encode()).hexdigest()[:16]


def window_day(day: date) -> int:
    """Index of ``day`` in the window (0 for the first day)."""
    return (day - WINDOW_START).days


def window_date(index: int) -> date:
    return WINDOW_START + timedelta(days=index)
