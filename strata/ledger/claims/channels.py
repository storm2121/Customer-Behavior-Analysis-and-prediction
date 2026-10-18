"""Claims about who uses which channel, and who goes silent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strata.contracts.vocab import age_bands, primary_cutoff
from strata.ledger.bootstrap import percentile_interval, weighted_quantile
from strata.ledger.claims._common import activity_tercile, require, shares, slug
from strata.ledger.model import Unit
from strata.ledger.registry import ClaimContext, ClaimResult, claim
from strata.ledger.robustness import RobustnessSpec

PROFILES = ("none", "branch_only", "digital_only", "mixed")
PROFILE_DEFINITION = (
    "Channel profile over the rolling year: none (no event), branch_only (branch operations only), "
    "digital_only (card or online, never the branch), mixed (branch and at least one digital "
    "channel)."
)


def _profile_matrix(profiles: pd.Series) -> np.ndarray:
    return np.column_stack(
        [(profiles == profile).to_numpy(dtype=np.float64) for profile in PROFILES]
    )


@claim(
    "channel_mix",
    title="How clients split across channels",
    question="Which share of clients uses only the branch, only digital channels, both, or none?",
    method=(
        "Whole-year channel profile of every client of the sample (client_360); shares with "
        "percentile intervals from a client bootstrap."
    ),
)
def channel_mix(context: ClaimContext) -> ClaimResult:
    clients = context.table("marts.client_360")
    require(len(clients) > 0, "no clients")
    boot = context.bootstrap(len(clients))
    values, intervals = shares(
        boot, _profile_matrix(clients["channel_profile"]), [f"share_{p}" for p in PROFILES]
    )
    n = len(clients)
    return ClaimResult(
        values=values,
        units=dict.fromkeys(values, Unit.SHARE),
        ci=intervals,
        value_n=dict.fromkeys(values, n),
        n=n,
        meta={"profile_definition": PROFILE_DEFINITION},
        caveats=[
            "A profile summarises the whole year: one branch visit makes a client mixed.",
            "Channels are the three files: ATM withdrawals and card payments count as card.",
        ],
    )


def _age_contrast(age: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(age >= 55, "55+", np.where(age < 35, "under 35", "35-54")), index=age.index
    )


@claim(
    "age_digital_divide",
    title="The digital divide by age",
    question=(
        "How do branch-only and digital-only shares change with age, and is there a gender gap?"
    ),
    method=(
        "Share of branch-only and digital-only clients in each age band (client_360, all clients "
        "of the band as denominator); gender gap as the difference of branch-only shares between "
        "the two gender codes; client bootstrap intervals. Headline robustness: branch-only share "
        "of clients aged 55 and over against clients under 35, within gender and tenure strata."
    ),
    headline=True,
    robustness=RobustnessSpec(
        group_col="age_contrast",
        treatment="55+",
        reference="under 35",
        metric="is_branch_only",
        strata=("gender", "tenure_band"),
        metric_label="branch-only share",
        treatment_label="clients aged 55 and over",
        reference_label="clients under 35",
    ),
)
def age_digital_divide(context: ClaimContext) -> ClaimResult:
    clients = context.table("marts.client_360")
    require(len(clients) > 0, "no clients")
    boot = context.bootstrap(len(clients))
    branch_only = (clients["channel_profile"] == "branch_only").to_numpy(dtype=np.float64)
    digital_only = (clients["channel_profile"] == "digital_only").to_numpy(dtype=np.float64)
    values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    value_n: dict[str, int] = {}
    meta: dict[str, str] = {}
    for band in age_bands():
        inside = (clients["age_band"] == band.label).to_numpy(dtype=np.float64)
        size = int(inside.sum())
        if size == 0:
            continue
        key = slug(band.label)
        meta[f"band_{key}"] = band.label
        for name, indicator in (("branch_only", branch_only), ("digital_only", digital_only)):
            value_key = f"{name}_{key}"
            values[value_key] = float((indicator * inside).sum() / size)
            intervals[value_key] = percentile_interval(boot.ratio(indicator * inside, inside))
            value_n[value_key] = size
    gender_one = (clients["gender_code"] == 1).to_numpy(dtype=np.float64)
    gender_two = (clients["gender_code"] == 2).to_numpy(dtype=np.float64)
    require(gender_one.sum() > 0 and gender_two.sum() > 0, "both gender codes are needed")
    gap = (branch_only * gender_two).sum() / gender_two.sum() - (
        branch_only * gender_one
    ).sum() / gender_one.sum()
    values["branch_only_gender_gap"] = float(gap)
    intervals["branch_only_gender_gap"] = percentile_interval(
        boot.ratio(branch_only * gender_two, gender_two)
        - boot.ratio(branch_only * gender_one, gender_one)
    )
    value_n["branch_only_gender_gap"] = int(min(gender_one.sum(), gender_two.sum()))
    units = {key: Unit.DIFFERENCE if key.endswith("gender_gap") else Unit.SHARE for key in values}
    meta["rows_bands"] = "|".join(
        f"{band.label}:{slug(band.label)}"
        for band in age_bands()
        if f"band_{slug(band.label)}" in meta
    )
    meta["contrast"] = "clients aged 55 and over against clients under 35"
    meta["profile_definition"] = PROFILE_DEFINITION
    frame = pd.DataFrame(
        {
            "age_contrast": _age_contrast(clients["age"]),
            "is_branch_only": branch_only,
            "gender": clients["gender_code"].astype("string").fillna("missing"),
            "tenure_band": clients["tenure_band"].astype("string").fillna("missing"),
        }
    )
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        value_n=value_n,
        n=len(clients),
        meta=meta,
        caveats=[
            "Age is measured at extraction, not when the channel habits formed.",
            "Shares use every client of the band as denominator, including clients with no event.",
            "Gender codes are kept as codes; their meaning is assumed, not documented.",
            (
                "The robustness check leaves activity level out: activity is partly a consequence "
                "of channel choice, so holding it fixed would bias the comparison."
            ),
        ],
        robustness_frame=frame,
    )


@claim(
    "online_service_mix",
    title="What the online channel is used for",
    question="Which services make up online operations, and how large is a typical bill payment?",
    method=(
        "Share of online operations by service (int_events), median amount of bill-payment "
        "operations, and the share of zero-amount operations that are card-service operations; "
        "client bootstrap intervals."
    ),
)
def online_service_mix(context: ClaimContext) -> ClaimResult:
    events = context.table("intermediate.int_events")
    online = events[events["channel"] == "online"].reset_index(drop=True)
    require(len(online) > 0, "no online operations")
    services = sorted(online["kind"].dropna().unique())
    codes = pd.Index(pd.unique(online["client_id"])).get_indexer(pd.Index(online["client_id"]))
    n_clients = int(codes.max()) + 1
    boot = context.bootstrap(n_clients)
    matrix = np.zeros((n_clients, len(services)))
    np.add.at(
        matrix, (codes, online["kind"].map({s: i for i, s in enumerate(services)}).to_numpy()), 1.0
    )
    values, intervals = shares(boot, matrix, [f"share_{s}" for s in services])
    bills = online["kind"] == "bill_pay"
    if bills.any():
        amounts = online.loc[bills, "amount"].to_numpy(dtype=np.float64)
        bill_codes = codes[bills.to_numpy()]
        values["median_bill_pay_amount"] = float(np.nanmedian(amounts))
        draws = [weighted_quantile(amounts, weights[bill_codes], 0.5) for weights in boot.weights]
        intervals["median_bill_pay_amount"] = percentile_interval(draws)
    zero = online["amount"] == 0
    values["zero_amount_card_service_share"] = (
        float((online.loc[zero, "kind"] == "card_service").mean()) if zero.any() else 1.0
    )
    units = {key: Unit.AMOUNT if key.startswith("median") else Unit.SHARE for key in values}
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        value_n=dict.fromkeys(values, n_clients),
        n=n_clients,
        n_events=len(online),
        caveats=[
            "Bill payment, mobile top-up and vehicle tax share one service label in the extract."
        ],
    )


@claim(
    "branch_deposit_share",
    title="Branches take deposits; cards give cash",
    question="Which share of branch operations are cash deposits, and where do withdrawals happen?",
    method=(
        "Share of branch operations in the deposit category and share of credits among branch "
        "operations; share of withdrawal value made by card rather than at the branch "
        "(int_events); client bootstrap intervals."
    ),
)
def branch_deposit_share(context: ClaimContext) -> ClaimResult:
    events = context.table("intermediate.int_events")
    branch = events[events["channel"] == "branch"]
    require(len(branch) > 0, "no branch operations")
    clients = pd.Index(pd.unique(events["client_id"]))
    boot = context.bootstrap(len(clients))

    def per_client(mask: pd.Series, *, amounts: bool = False) -> np.ndarray:
        selected = events[mask]
        index = clients.get_indexer(pd.Index(selected["client_id"]))
        weights = selected["amount"].fillna(0).to_numpy() if amounts else np.ones(len(selected))
        return np.bincount(index, weights=weights, minlength=len(clients)).astype(np.float64)

    is_branch = events["channel"] == "branch"
    branch_count = per_client(is_branch)
    deposits = per_client(is_branch & (events["kind_group"] == "deposit"))
    credit_rows = per_client(is_branch & (events["direction"] == "credit"))
    card_withdrawals = per_client(
        (events["channel"] == "card") & (events["kind"] == "withdrawal"), amounts=True
    )
    branch_withdrawals = per_client(
        is_branch & (events["kind_group"] == "withdrawal"), amounts=True
    )
    total_withdrawals = card_withdrawals + branch_withdrawals
    values = {
        "deposit_share_of_branch_operations": float(deposits.sum() / branch_count.sum()),
        "credit_share_of_branch_operations": float(credit_rows.sum() / branch_count.sum()),
        "card_share_of_withdrawal_value": float(card_withdrawals.sum() / total_withdrawals.sum())
        if total_withdrawals.sum() > 0
        else 0.0,
    }
    intervals = {
        "deposit_share_of_branch_operations": percentile_interval(
            boot.ratio(deposits, branch_count)
        ),
        "credit_share_of_branch_operations": percentile_interval(
            boot.ratio(credit_rows, branch_count)
        ),
        "card_share_of_withdrawal_value": percentile_interval(
            boot.ratio(card_withdrawals, total_withdrawals)
        ),
    }
    n_branch = int((branch_count > 0).sum())
    return ClaimResult(
        values=values,
        units=dict.fromkeys(values, Unit.SHARE),
        ci=intervals,
        value_n=dict.fromkeys(values, n_branch),
        n=n_branch,
        n_events=len(branch),
        caveats=[
            (
                "Deposit and withdrawal categories come from the operation labels (seed); rare "
                "labels have assumed meanings."
            ),
        ],
    )


@claim(
    "silence_by_channel",
    title="Who goes silent: branch-only against digital-only clients",
    question=(
        "Are branch-only clients more likely than digital-only clients to have no event in the "
        "quarter after the observation window, and does that survive holding activity fixed?"
    ),
    method=(
        "Universe: clients with at least one event in the observation window of the primary cutoff "
        "(features_silence). Silent: no event in the outcome window. Silence rate by channel "
        "profile, risk difference and risk ratio of branch-only against digital-only, client "
        "bootstrap intervals; Simpson guard within activity terciles, age bands and tenure bands."
    ),
    headline=True,
    robustness=RobustnessSpec(
        group_col="channel_profile",
        treatment="branch_only",
        reference="digital_only",
        metric="silent",
        metric_label="silence rate",
        treatment_label="branch-only clients",
        reference_label="digital-only clients",
    ),
)
def silence_by_channel(context: ClaimContext) -> ClaimResult:
    cutoff = primary_cutoff()
    features = context.table("marts.features_silence")
    universe = features[features["cutoff_date"] == pd.Timestamp(cutoff.cutoff)].reset_index(
        drop=True
    )
    require(len(universe) > 0, "empty silence universe")
    boot = context.bootstrap(len(universe))
    silent = universe["silent"].to_numpy(dtype=np.float64)
    values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    value_n: dict[str, int] = {}
    indicators = {}
    for profile in ("branch_only", "digital_only", "mixed"):
        inside = (universe["channel_profile"] == profile).to_numpy(dtype=np.float64)
        require(inside.sum() > 0, f"no {profile} clients in the universe")
        indicators[profile] = inside
        values[f"silence_rate_{profile}"] = float((silent * inside).sum() / inside.sum())
        intervals[f"silence_rate_{profile}"] = percentile_interval(
            boot.ratio(silent * inside, inside)
        )
        value_n[f"silence_rate_{profile}"] = int(inside.sum())
        values[f"median_events_{profile}"] = float(universe.loc[inside > 0, "n_events"].median())
        value_n[f"median_events_{profile}"] = int(inside.sum())
    branch, digital = indicators["branch_only"], indicators["digital_only"]
    rate_branch = boot.ratio(silent * branch, branch)
    rate_digital = boot.ratio(silent * digital, digital)
    values["risk_difference"] = (
        values["silence_rate_branch_only"] - values["silence_rate_digital_only"]
    )
    intervals["risk_difference"] = percentile_interval(rate_branch - rate_digital)
    require(values["silence_rate_digital_only"] > 0, "no silent digital-only client")
    values["risk_ratio"] = values["silence_rate_branch_only"] / values["silence_rate_digital_only"]
    with np.errstate(divide="ignore", invalid="ignore"):
        intervals["risk_ratio"] = percentile_interval(rate_branch / rate_digital)
    value_n["risk_difference"] = value_n["risk_ratio"] = int(min(branch.sum(), digital.sum()))
    units = dict.fromkeys(values, Unit.SHARE)
    units.update({"risk_difference": Unit.DIFFERENCE, "risk_ratio": Unit.RATIO})
    units.update({key: Unit.NUMBER for key in values if key.startswith("median_events")})
    frame = pd.DataFrame(
        {
            "channel_profile": universe["channel_profile"],
            "silent": silent,
            "activity_tercile": activity_tercile(universe["n_events"]),
            "age_band": universe["age_band"].astype("string").fillna("missing"),
            "tenure_band": universe["tenure_band"].astype("string").fillna("missing"),
        }
    )
    hint = (
        "Mechanism: branch-only clients have a median of "
        f"{values['median_events_branch_only']:.0f} events in the observation window against "
        f"{values['median_events_digital_only']:.0f} for "
        "digital-only clients. With so few events a quarter without any is likely by chance, so "
        "the crude gap mostly measures activity, not channel."
    )
    # The same mechanism in words only, for the verdict-only publication; stated only when the
    # data support it.
    fewer = values["median_events_branch_only"] < values["median_events_digital_only"]
    public_hint = (
        (
            "Mechanism: branch-only clients have fewer events in the observation window than "
            "digital-only clients; with few events a quarter without any is likely by chance, so "
            "the crude gap mostly measures activity, not channel."
        )
        if fewer
        else ""
    )
    return ClaimResult(
        values=values,
        units=units,
        ci=intervals,
        value_n=value_n,
        n=len(universe),
        meta={
            "cutoff": cutoff.cutoff.isoformat(),
            "outcome_start": cutoff.outcome_start.isoformat(),
            "outcome_end": cutoff.outcome_end.isoformat(),
        },
        caveats=[
            "Silence is the absence of events in three months, not a closed account.",
            "Profiles are measured over the observation window only (no look-ahead).",
            (
                "Activity terciles where one group is absent cannot be compared (positivity); they "
                "are listed in the robustness result and excluded from the adjusted difference."
            ),
        ],
        robustness_frame=frame,
        mechanism_hint=hint,
        public_hint=public_hint,
    )
