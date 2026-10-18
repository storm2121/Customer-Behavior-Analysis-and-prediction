"""A differentially private synthetic twin of the client table (``strata --private synth-twin``).

The twin is a Gaussian copula fitted to ``marts.client_360`` with differential privacy on every
parameter it learns from real rows (ADR 0013). One client is one row, so the guarantee is per
client:

- **Marginals** are histograms over bins fixed in advance from public bounds and domains (never
  read from the data), noised with the Laplace mechanism: changing one client's row moves two
  counts of each column's histogram by one. Values are clipped to the public bounds first.
- **Dependence** is Kendall's tau-a between every pair of columns, noised with the Laplace
  mechanism (one client's row moves a tau-a by at most 4/(n-1)), mapped to a correlation with
  rho = sin(pi tau / 2) and projected onto the correlation matrices.
- **Sampling** the copula and reading the values back through the noisy marginals is
  post-processing, which costs no privacy.

The budget epsilon is split between the marginals and the correlations, each share divided evenly
among its columns or pairs. The number of clients is treated as public (the public claims publish
it). Missing values are a bin of their own, placed after the highest values, which the copula
treats as ordinal; categories are ordered by their public domain. Kendall's tau-a counts ties as
neither concordant nor discordant, so heavily tied columns come out less dependent than they are.

The twin is fitted on a random part of the clients; the rest is a holdout that the metrics compare
against. The metrics read real rows and are not private themselves, which is why they stay in the
private folder with the twin:

- fidelity: per column, the Kolmogorov-Smirnov distance (numbers) or the total-variation distance
  (categories) between twin and training rows; the mean absolute difference of their Spearman
  correlation matrices; and how well a logistic regression tells twin rows from real ones (AUC,
  one half when it cannot);
- privacy: the distance from each twin row to its closest training row against the same distance
  for holdout rows, and the share of twin rows closer than the holdout's fifth percentile (about
  five per cent when the twin does not copy); the share of binned twin rows equal to a binned
  training row against the holdout's share; and a distance-based membership-inference attack,
  whose AUC on training members against holdout non-members is one half when the twin carries no
  membership signal.

The twin carries no synthetic marker, so a public ingest refuses it like any unmarked extract.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors

from strata.config import PrivacyBoundaryError, Settings, Zone, is_within
from strata.ingest.load import load_manifest
from strata.warehouse import connect, read_relation

Kind = Literal["count", "amount", "share", "integer", "category"]

MISSING = "missing"
OTHER = "other"
ATOM = 1e-9
"""Half-width of the bins that hold a point mass (zero amounts, shares of exactly zero or one)."""
HISTOGRAM_SENSITIVITY = 2.0
"""Neighbouring tables differ in one client's row (the number of clients is public), which moves
one count of a histogram down by one and another up by one."""
TWIN_FILE = "twin.csv"
METRICS_FILE = "metrics.json"


class TwinError(ValueError):
    """The twin cannot be built as asked (wrong zone, too few clients, ...)."""


@dataclass(frozen=True, slots=True)
class Column:
    """One twin column: its kind and the public bounds or domain its bins are made from."""

    name: str
    kind: Kind
    low: float = 0.0
    high: float = 0.0
    step: float = 1.0
    domain: tuple[str, ...] = ()

    @property
    def numeric(self) -> bool:
        return self.kind != "category"

    def edges(self) -> np.ndarray:
        """Bin edges of a numeric column, fixed by its kind and public bounds.

        Values that pile up on one point get a bin of almost no width around it, so the twin
        reproduces the pile instead of spreading it over a bin: no amount (zero), a share of
        exactly zero or one. Counts and integers are floored, so their first bins are exact.
        """
        if self.kind == "count":
            tail = np.unique(np.round(np.geomspace(4, self.high + 1, 14)))
            return np.unique(np.concatenate([[0.0, 1.0, 2.0, 3.0], tail]))
        if self.kind == "amount":
            return np.concatenate([[0.0, ATOM], np.geomspace(1.0, self.high, 17)])
        if self.kind == "share":
            inner = np.linspace(0.0, 1.0, 11)[1:-1]
            return np.concatenate([[0.0, ATOM], inner, [1.0 - ATOM, 1.0 + ATOM]])
        if self.kind == "integer":
            starts = np.arange(self.low, self.high + 1, self.step, dtype=float)
            return np.unique(np.append(starts, self.high + 1))
        msg = f"{self.name} is a category and has no edges"
        raise TypeError(msg)

    @property
    def upper(self) -> float:
        """The largest value the column may take."""
        return 1.0 if self.kind == "share" else self.high

    def labels(self) -> tuple[str, ...]:
        """The categories of a category column, then the catch-alls."""
        return (*self.domain, OTHER, MISSING)

    def n_bins(self) -> int:
        """Bins including the missing one."""
        return len(self.labels()) if not self.numeric else len(self.edges())


COLUMNS: tuple[Column, ...] = (
    Column("age", "integer", low=18, high=100, step=5),
    Column("tenure_years", "integer", low=0, high=60, step=5),
    Column("gender", "category", domain=("gender_1", "gender_2")),
    Column(
        "marital_status", "category", domain=("marital_1", "marital_2", "marital_3", "marital_4")
    ),
    Column("channel_profile", "category", domain=("none", "branch_only", "digital_only", "mixed")),
    Column("active_months", "integer", low=0, high=12),
    Column("n_events", "count", high=20_000),
    Column("n_card", "count", high=20_000),
    Column("n_online", "count", high=20_000),
    Column("n_branch", "count", high=20_000),
    Column("card_spend", "amount", high=1e8),
    Column("branch_credit_amount", "amount", high=1e8),
    Column("online_amount", "amount", high=1e8),
    Column("branch_share", "share"),
)
"""The client columns the twin reproduces; identifiers and dates are never among them."""


@dataclass(frozen=True, slots=True)
class TwinConfig:
    epsilon: float = 1.0
    marginal_share: float = 0.5
    holdout: float = 0.2
    rows: int = 1000
    seed: int = 42
    min_clients: int = 200

    def __post_init__(self) -> None:
        if not 0 < self.epsilon <= 100:
            msg = "epsilon must be in (0, 100]"
            raise TwinError(msg)
        if not 0 < self.marginal_share < 1 or not 0 < self.holdout < 1:
            msg = "marginal_share and holdout must be strictly between 0 and 1"
            raise TwinError(msg)


@dataclass(frozen=True, slots=True)
class Copula:
    """The fitted twin: noisy bin probabilities per column and a correlation matrix."""

    columns: tuple[Column, ...]
    probabilities: tuple[np.ndarray, ...]
    correlation: np.ndarray


# ---------------------------------------------------------------------------------------------
# Encoding real rows into bins and ranks


def bin_index(column: Column, values: pd.Series) -> np.ndarray:
    """The bin of every value: numeric bins (after clipping to the bounds) or categories, with
    missing values in the last bin."""
    n_bins = column.n_bins()
    if column.numeric:
        numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        missing = np.isnan(numbers)
        edges = column.edges()
        clipped = np.clip(np.nan_to_num(numbers, nan=edges[0]), edges[0], edges[-1])
        index = np.clip(np.searchsorted(edges, clipped, side="right") - 1, 0, n_bins - 2)
        return np.where(missing, n_bins - 1, index)
    labels = column.labels()
    lookup = {label: position for position, label in enumerate(column.domain)}
    text = values.astype("string")
    out = np.array([lookup.get(str(v), len(column.domain)) for v in text.fillna(MISSING)])
    return np.where(text.isna().to_numpy(), len(labels) - 1, out)


def rank_values(column: Column, values: pd.Series) -> np.ndarray:
    """Values in the order the copula uses: numbers (missing after the largest) or category
    positions in the public domain."""
    if column.numeric:
        numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        edges = column.edges()
        clipped = np.clip(numbers, edges[0], edges[-1])
        return np.asarray(np.where(np.isnan(numbers), edges[-1] + 1.0, clipped), dtype=np.float64)
    return bin_index(column, values).astype(np.float64)


def kendall_tau_a(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-a: (concordant - discordant) / all pairs, ties counting as neither."""
    n = len(x)
    if n < 2:
        return 0.0
    pairs = n * (n - 1) / 2
    tau_b = stats.kendalltau(x, y, variant="b").statistic
    if not math.isfinite(tau_b):
        return 0.0

    def tied(values: np.ndarray) -> float:
        counts = np.unique(values, return_counts=True)[1].astype(np.float64)
        return float((counts * (counts - 1) / 2).sum())

    scale = math.sqrt((pairs - tied(x)) * (pairs - tied(y))) / pairs
    return float(tau_b * scale)


def tau_sensitivity(n: int) -> float:
    """How far one client's row can move Kendall's tau-a between two columns of ``n`` rows: the
    row takes part in ``n - 1`` of the ``n (n - 1) / 2`` pairs, each moving by at most two."""
    return 4.0 / max(n - 1, 1)


def nearest_correlation(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """The matrix with its eigenvalues raised to ``floor`` and its diagonal rescaled to ones."""
    symmetric = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    fixed = (vectors * np.maximum(values, floor)) @ vectors.T
    scale = np.sqrt(np.diag(fixed))
    out = fixed / np.outer(scale, scale)
    np.fill_diagonal(out, 1.0)
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# Fitting with differential privacy, and sampling


def fit(
    frame: pd.DataFrame, columns: tuple[Column, ...], config: TwinConfig, rng: np.random.Generator
) -> Copula:
    """The copula of ``frame`` with Laplace noise on every histogram count and every tau."""
    n = len(frame)
    d = len(columns)
    epsilon_marginals = config.epsilon * config.marginal_share
    epsilon_pairs = config.epsilon - epsilon_marginals
    per_column = epsilon_marginals / d
    probabilities = []
    for column in columns:
        counts = np.bincount(bin_index(column, frame[column.name]), minlength=column.n_bins())
        noise = rng.laplace(0.0, HISTOGRAM_SENSITIVITY / per_column, size=counts.size)
        noisy = np.maximum(counts + noise, 0.0)
        total = noisy.sum()
        probabilities.append(
            noisy / total if total > 0 else np.full(counts.size, 1.0 / counts.size)
        )
    ranks = [rank_values(column, frame[column.name]) for column in columns]
    n_pairs = d * (d - 1) // 2
    scale = n_pairs * tau_sensitivity(n) / epsilon_pairs if n_pairs else 0.0
    correlation = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau = kendall_tau_a(ranks[i], ranks[j]) + rng.laplace(0.0, scale)
            correlation[i, j] = correlation[j, i] = math.sin(
                math.pi * float(np.clip(tau, -1, 1)) / 2
            )
    return Copula(columns, tuple(probabilities), nearest_correlation(correlation))


def _decode(column: Column, probabilities: np.ndarray, u: np.ndarray) -> pd.Series:
    """Values of one column from uniforms, through its noisy bin probabilities."""
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    index = np.clip(np.searchsorted(cumulative, u, side="right"), 0, probabilities.size - 1)
    if not column.numeric:
        labels = pd.Series(np.asarray(column.labels(), dtype=object)[index], dtype="string")
        return labels.mask(index == probabilities.size - 1)
    start = np.concatenate([[0.0], cumulative[:-1]])[index]
    width = np.maximum(probabilities[index], 1e-12)
    fraction = np.clip((u - start) / width, 0.0, 1.0 - 1e-9)
    edges = column.edges()
    last = probabilities.size - 1
    low = edges[np.minimum(index, last - 1)]
    high = edges[np.minimum(index + 1, last)]
    values = np.clip(low + fraction * (high - low), edges[0], column.upper)
    if column.kind in {"count", "integer"}:
        values = np.floor(values)
    return pd.Series(np.where(index == last, np.nan, values), dtype="Float64")


def sample(copula: Copula, rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """``rows`` twin rows: correlated normals, their uniforms, and the noisy marginals."""
    factor = np.linalg.cholesky(copula.correlation)
    normals = rng.standard_normal((rows, len(copula.columns))) @ factor.T
    uniforms = stats.norm.cdf(normals)
    return pd.DataFrame(
        {
            column.name: _decode(column, probabilities, uniforms[:, position])
            for position, (column, probabilities) in enumerate(
                zip(copula.columns, copula.probabilities, strict=True)
            )
        }
    )


# ---------------------------------------------------------------------------------------------
# Fidelity and privacy metrics (on real rows: private outputs)


def _embedding(
    columns: tuple[Column, ...], frame: pd.DataFrame, reference: pd.DataFrame
) -> np.ndarray:
    """Rows as points: numbers through the reference's empirical distribution (missing at -0.25),
    categories one-hot scaled so that a mismatch weighs like a numeric distance of one."""
    parts = []
    for column in columns:
        if column.numeric:
            base = np.sort(
                pd.to_numeric(reference[column.name], errors="coerce").dropna().to_numpy()
            )
            values = pd.to_numeric(frame[column.name], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            position = np.searchsorted(base, np.nan_to_num(values), side="right") / max(
                len(base), 1
            )
            parts.append(np.where(np.isnan(values), -0.25, position)[:, None])
        else:
            index = bin_index(column, frame[column.name])
            onehot = np.zeros((len(frame), column.n_bins()))
            onehot[np.arange(len(frame)), index] = math.sqrt(0.5)
            parts.append(onehot)
    return np.hstack(parts)


def _distribution_distance(column: Column, twin: pd.Series, real: pd.Series) -> float:
    if column.numeric:
        a = (
            pd.to_numeric(twin, errors="coerce")
            .dropna()
            .to_numpy(dtype=np.float64, na_value=np.nan)
        )
        b = (
            pd.to_numeric(real, errors="coerce")
            .dropna()
            .to_numpy(dtype=np.float64, na_value=np.nan)
        )
        if len(a) == 0 or len(b) == 0:
            return 1.0
        return float(stats.ks_2samp(a, b).statistic)
    p = np.bincount(bin_index(column, twin), minlength=column.n_bins()) / max(len(twin), 1)
    q = np.bincount(bin_index(column, real), minlength=column.n_bins()) / max(len(real), 1)
    return float(0.5 * np.abs(p - q).sum())


def _spearman(columns: tuple[Column, ...], frame: pd.DataFrame) -> np.ndarray:
    ranks = np.column_stack([rank_values(column, frame[column.name]) for column in columns])
    matrix = stats.spearmanr(ranks).statistic
    return np.nan_to_num(np.atleast_2d(matrix), nan=0.0)


def _distinguishability(points_real: np.ndarray, points_twin: np.ndarray, seed: int) -> float:
    n = min(len(points_real), len(points_twin), 20_000)
    rng = np.random.default_rng(seed)
    real = points_real[rng.choice(len(points_real), n, replace=False)]
    twin = points_twin[rng.choice(len(points_twin), n, replace=False)]
    x = np.vstack([real, twin])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_predict(model, x, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, scores))


def metrics(
    columns: tuple[Column, ...],
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    twin: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    """Fidelity of the twin to the training rows, and what it reveals about them."""
    points_train = _embedding(columns, train, train)
    points_holdout = _embedding(columns, holdout, train)
    points_twin = _embedding(columns, twin, train)

    to_train = NearestNeighbors(n_neighbors=1).fit(points_train)
    twin_dcr = to_train.kneighbors(points_twin)[0][:, 0]
    holdout_dcr = to_train.kneighbors(points_holdout)[0][:, 0]
    threshold = float(np.quantile(holdout_dcr, 0.05))

    rng = np.random.default_rng(seed)
    members = points_train[rng.choice(len(points_train), len(points_holdout), replace=False)]
    to_twin = NearestNeighbors(n_neighbors=1).fit(points_twin)
    member_distance = to_twin.kneighbors(members)[0][:, 0]
    outsider_distance = to_twin.kneighbors(points_holdout)[0][:, 0]
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(points_holdout))])
    scores = -np.concatenate([member_distance, outsider_distance])
    fpr, tpr, _ = roc_curve(labels, scores)

    def binned(frame: pd.DataFrame) -> set[tuple[int, ...]]:
        codes = np.column_stack([bin_index(column, frame[column.name]) for column in columns])
        return {tuple(int(v) for v in row) for row in codes}

    train_cells = binned(train)

    def matched(frame: pd.DataFrame) -> float:
        codes = np.column_stack([bin_index(column, frame[column.name]) for column in columns])
        return float(np.mean([tuple(int(v) for v in row) in train_cells for row in codes]))

    gap = np.abs(_spearman(columns, twin) - _spearman(columns, train))
    upper = gap[np.triu_indices(len(columns), k=1)]
    return {
        "fidelity": {
            "marginal_distance": {
                column.name: _distribution_distance(column, twin[column.name], train[column.name])
                for column in columns
            },
            "marginal_measure": {
                column.name: "kolmogorov_smirnov" if column.numeric else "total_variation"
                for column in columns
            },
            "spearman_mean_abs_difference": float(upper.mean()) if upper.size else 0.0,
            "spearman_max_abs_difference": float(upper.max()) if upper.size else 0.0,
            "distinguishability_auc": _distinguishability(points_train, points_twin, seed),
        },
        "privacy": {
            "dcr_twin_median": float(np.median(twin_dcr)),
            "dcr_holdout_median": float(np.median(holdout_dcr)),
            "dcr_ratio": float(np.median(twin_dcr) / max(float(np.median(holdout_dcr)), 1e-12)),
            "share_twin_closer_than_holdout_p05": float(np.mean(twin_dcr < threshold)),
            "binned_match_twin": matched(twin),
            "binned_match_holdout": matched(holdout),
            "membership_inference_auc": float(roc_auc_score(labels, scores)),
            "membership_inference_advantage": float(np.max(tpr - fpr)),
        },
    }


# ---------------------------------------------------------------------------------------------
# The command


def check_output(settings: Settings, out: Path) -> Path:
    """Refuse, before anything is read, a twin bound for CI, the repository or outside the
    private folder."""
    if os.environ.get("CI"):
        msg = "synth-twin runs only on the machine that holds the extract, never in CI."
        raise PrivacyBoundaryError(msg)
    resolved = out.expanduser().resolve()
    if is_within(resolved, settings.repo_root):
        msg = "Refusing to write the twin inside the repository."
        raise PrivacyBoundaryError(msg)
    return settings.assert_private_output(resolved)


def build(frame: pd.DataFrame, config: TwinConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Split, fit with differential privacy, sample, and measure: the twin and its metrics."""
    if len(frame) < config.min_clients:
        msg = f"the twin needs at least {config.min_clients} clients"
        raise TwinError(msg)
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(frame))
    cut = round(len(frame) * (1 - config.holdout))
    train = frame.iloc[order[:cut]].reset_index(drop=True)
    holdout = frame.iloc[order[cut:]].reset_index(drop=True)
    copula = fit(train, COLUMNS, config, rng)
    twin = sample(copula, config.rows, rng)
    measured = metrics(COLUMNS, train, holdout, twin, config.seed)
    summary = {
        "epsilon": config.epsilon,
        "epsilon_marginals": config.epsilon * config.marginal_share,
        "epsilon_correlations": config.epsilon * (1 - config.marginal_share),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "n_twin": len(twin),
        "columns": [asdict(column) for column in COLUMNS],
        **measured,
    }
    return twin, summary


def run(settings: Settings, zone: Zone, out: Path, config: TwinConfig) -> list[Path]:
    """The twin of the private zone's clients, written with its metrics under ``out``."""
    target = check_output(settings, out)
    if zone is not Zone.PRIVATE:
        msg = "The twin is made from the real extract: run `strata --private synth-twin`."
        raise TwinError(msg)
    workspace = settings.workspace(Zone.PRIVATE)
    manifest = load_manifest(workspace)
    con = connect(workspace.warehouse, read_only=True)
    try:
        clients = read_relation(con, "marts.client_360")
    finally:
        con.close()
    twin, summary = build(clients[[column.name for column in COLUMNS]], config)
    target.mkdir(parents=True, exist_ok=True)
    twin_path = target / TWIN_FILE
    metrics_path = target / METRICS_FILE
    twin.to_csv(twin_path, index=False, lineterminator="\n")
    summary = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "data_fingerprint": manifest.fingerprint,
        "mapping_sha256": manifest.mapping_sha256,
        **summary,
    }
    metrics_path.write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return [twin_path, metrics_path]
