"""Behavioural segments of active clients (ARCHITECTURE section 10).

Features come from ``marts.features_segments`` (per-channel recency, frequency and value, channel
shares, international and e-commerce shares, deposit intensity, card-service operations,
round-amount share). Counts, amounts and recencies are log1p-transformed, missing recencies (a
channel never used) are set to the window length plus a flag, and everything is standardised.

- Gaussian mixtures with full covariances for k from two to ten; k is chosen by BIC.
- KMeans for the same k range with the silhouette score, as a cross-check.
- Stability: the adjusted Rand index between the chosen mixture's labels and those of mixtures
  refitted on bootstrap resamples of clients (labels compared on the clients each resample drew).
- Profiles: per-segment medians and shares, Kruskal-Wallis tests for features and chi-square tests
  for the channel profile and the age band; each segment is named after its two most distinctive
  features (largest absolute standardised median against the whole population).
- UMAP gives a two-dimensional embedding for display only (the first two principal components at
  quick effort); it plays no part in the segmentation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

RECENCIES = ("card_recency_days", "online_recency_days", "branch_recency_days")
LOGGED = (
    "card_frequency",
    "card_monetary",
    "online_frequency",
    "online_monetary",
    "branch_frequency",
    "branch_monetary",
    "deposits_per_active_month",
    "n_card_service_ops",
    *RECENCIES,
)
SHARES = (
    "card_share",
    "online_share",
    "branch_share",
    "international_share",
    "ecommerce_share",
    "round_amount_share",
)
FEATURES = (*LOGGED, *SHARES)
WINDOW_DAYS = 366


@dataclass(frozen=True)
class Effort:
    k_max: int
    resamples: int
    umap: bool


FULL = Effort(k_max=10, resamples=50, umap=True)
QUICK = Effort(k_max=5, resamples=8, umap=False)


def matrix(features: pd.DataFrame) -> pd.DataFrame:
    """The standardised feature matrix (one row per client, in client order)."""
    frame = pd.DataFrame(index=features.index)
    for name in FEATURES:
        values = pd.to_numeric(features[name], errors="coerce").astype(float)
        if name in RECENCIES:
            frame[f"{name.removesuffix('_days')}_never"] = values.isna().astype(float)
            values = values.fillna(WINDOW_DAYS)
        values = values.fillna(0.0)
        frame[name] = np.log1p(values.clip(lower=0)) if name in LOGGED else values
    scale = frame.std().replace(0, 1.0)
    return (frame - frame.mean()) / scale


def _mixture(x: np.ndarray, k: int, seed: int) -> GaussianMixture:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GaussianMixture(
            n_components=k, covariance_type="full", reg_covar=1e-4, n_init=2, random_state=seed
        ).fit(x)


def select(x: np.ndarray, effort: Effort, seed: int) -> dict[str, Any]:
    bic = {k: float(_mixture(x, k, seed).bic(x)) for k in range(2, effort.k_max + 1)}
    silhouette = {}
    for k in range(2, effort.k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
        silhouette[k] = float(silhouette_score(x, labels))
    return {"bic": bic, "silhouette": silhouette, "k": min(bic, key=lambda k: bic[k])}


def stability(
    x: np.ndarray, labels: np.ndarray, k: int, effort: Effort, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(effort.resamples):
        rows = rng.integers(0, len(x), size=len(x))
        refit = _mixture(x[rows], k, int(rng.integers(0, 2**31)))
        scores.append(adjusted_rand_score(labels[rows], refit.predict(x[rows])))
    low, high = np.percentile(scores, [2.5, 97.5])
    return {"mean_ari": float(np.mean(scores)), "ci_low": float(low), "ci_high": float(high)}


def describe(feature: str, *, high: bool) -> str:
    """Words for one distinctive feature of a segment (higher or lower than the population)."""
    channel = feature.split("_", 1)[0]
    if feature.endswith("_never"):
        return f"no {channel} use" if high else f"uses {channel}"
    if feature.endswith("_recency_days"):
        return f"{channel} use lapsed" if high else f"recent {channel} use"
    readable = feature.replace("_", " ")
    return f"{'high' if high else 'low'} {readable}"


def profiles(
    features: pd.DataFrame, x: pd.DataFrame, labels: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Per-segment summaries and names, and the tests of difference between segments."""
    segments = []
    overall = x.median()
    for segment in sorted(np.unique(labels)):
        members = labels == segment
        distinct = (x[members].median() - overall).abs().sort_values(ascending=False)
        top = list(distinct.index[:2])
        name = " / ".join(
            describe(feature, high=bool(x[members][feature].median() > overall[feature]))
            for feature in top
        )
        segments.append(
            {
                "segment": int(segment),
                "name": name,
                "n_clients": int(members.sum()),
                "share": float(members.mean()),
                "medians": {
                    f: float(
                        pd.to_numeric(features.loc[members, f], errors="coerce")
                        .astype(float)
                        .median()
                    )
                    for f in FEATURES
                },
                "channel_profile": features.loc[members, "channel_profile"]
                .value_counts(normalize=True)
                .to_dict(),
            }
        )
    tests: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        groups = [x.loc[labels == s, feature] for s in np.unique(labels)]
        if all(len(g) > 0 for g in groups) and len(groups) > 1:
            statistic, p = stats.kruskal(*groups)
            tests[feature] = {"kruskal_h": float(statistic), "p_value": float(p)}
    for categorical in ("channel_profile", "age_band"):
        table = pd.crosstab(labels, features[categorical].fillna("missing"))
        if table.shape[0] > 1 and table.shape[1] > 1:
            chi2, p, _, _ = stats.chi2_contingency(table)
            tests[categorical] = {"chi2": float(chi2), "p_value": float(p)}
    return segments, tests


def principal_plane(x: np.ndarray) -> np.ndarray:
    """The first two principal components (the quick stand-in for UMAP, also display only)."""
    centred = x - x.mean(axis=0)
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    return np.asarray(centred @ vectors[:2].T, dtype=np.float64)


def embedding(x: np.ndarray, seed: int) -> np.ndarray:
    """UMAP coordinates for display (two dimensions)."""
    import umap  # noqa: PLC0415 - heavy import, display only

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_components=2, random_state=seed, n_jobs=1)
        return np.asarray(reducer.fit_transform(x), dtype=np.float64)


def run(features: pd.DataFrame, *, effort: Effort = FULL, seed: int = 42) -> dict[str, Any]:
    rows = features.sort_values("client_id").reset_index(drop=True)
    x = matrix(rows)
    values = x.to_numpy()
    chosen = select(values, effort, seed)
    k = int(chosen["k"])
    model = _mixture(values, k, seed)
    labels = model.predict(values)
    segments, tests = profiles(rows, x, labels)
    assignments = pd.DataFrame({"client_id": rows["client_id"], "segment": labels})
    coordinates = embedding(values, seed) if effort.umap else principal_plane(values)
    assignments["embed_x"], assignments["embed_y"] = coordinates[:, 0], coordinates[:, 1]
    return {
        "k": k,
        "embedding": "umap" if effort.umap else "pca",
        "bic": chosen["bic"],
        "silhouette": chosen["silhouette"],
        "stability": stability(values, labels, k, effort, seed),
        "segments": segments,
        "tests": tests,
        "n_clients": len(rows),
        "assignments": assignments,
    }
