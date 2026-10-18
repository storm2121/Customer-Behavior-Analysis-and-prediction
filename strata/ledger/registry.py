"""The claim registry: ``@claim`` turns a function of the warehouse into a ledger entry.

A claim function receives a :class:`ClaimContext` (read-only access to the warehouse and a seeded
bootstrap factory) and returns a :class:`ClaimResult`. Registration records what the claim asks, how
it answers, whether it is a headline, and how its headline should be stress-tested; the ledger build
then adds provenance (fingerprint, time, version) and runs the robustness checker.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import duckdb
import pandas as pd

from strata.ledger.bootstrap import DEFAULT_RESAMPLES, Bootstrap, claim_seed
from strata.ledger.model import Unit
from strata.ledger.robustness import RobustnessSpec
from strata.warehouse import read_relation

MODEL_RESAMPLES = 200
QUICK_MODEL_RESAMPLES = 30
POSTERIOR_SAMPLES = 2000
QUICK_POSTERIOR_SAMPLES = 500
"""Effort settings. Full runs use them all; ``quick`` runs (tests, demos) use the smaller ones, the
build marks every quick claim with ``meta["effort"] = "quick"`` and ``--public`` refuses quick
builds, so a quick number is never mistaken for a full one."""


class ClaimError(RuntimeError):
    """A claim cannot be computed on this warehouse (for example, too few clients in a group)."""


@dataclass(frozen=True)
class ClaimResult:
    """What a claim function returns; provenance is added by the build."""

    values: dict[str, float]
    units: dict[str, Unit]
    n: int
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    interval_label: str = "95% CI"
    value_n: dict[str, int] = field(default_factory=dict)
    n_events: int | None = None
    caveats: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    robustness_frame: pd.DataFrame | None = None
    mechanism_hint: str = ""
    public_hint: str = ""
    """A number-free completion of the explanation, for ``verdict_only`` publication."""


class ClaimContext:
    """Read-only access to one warehouse, plus seeded bootstraps."""

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        seed: int = 42,
        n_resamples: int = DEFAULT_RESAMPLES,
        quick: bool = False,
    ) -> None:
        self._con = con
        self.seed = seed
        self.n_resamples = n_resamples
        self.quick = quick
        self._cache: dict[str, pd.DataFrame] = {}
        self.claim_id = ""

    @property
    def model_resamples(self) -> int:
        """Client-bootstrap refits for model-based claims (each refit fits a regression)."""
        return min(self.n_resamples, QUICK_MODEL_RESAMPLES if self.quick else MODEL_RESAMPLES)

    @property
    def posterior_samples(self) -> int:
        """Posterior draws for the Markov chain's transition matrix."""
        return QUICK_POSTERIOR_SAMPLES if self.quick else POSTERIOR_SAMPLES

    def table(self, name: str) -> pd.DataFrame:
        """A whole warehouse table (``schema.table``), cached for the duration of a build."""
        if name not in self._cache:
            self._cache[name] = read_relation(self._con, name)
        return self._cache[name].copy()

    def bootstrap(
        self, n_clients: int, stream: str = "", resamples: int | None = None
    ) -> Bootstrap:
        """A bootstrap seeded by the master seed, the current claim's id and an optional stream."""
        seed = claim_seed(self.seed, f"{self.claim_id}/{stream}" if stream else self.claim_id)
        return Bootstrap(n_clients, n_resamples=resamples or self.n_resamples, seed=seed)


ClaimFunction = Callable[[ClaimContext], ClaimResult]


@dataclass(frozen=True)
class ClaimSpec:
    """A registered claim."""

    id: str
    title: str
    question: str
    method: str
    compute: ClaimFunction
    headline: bool = False
    robustness: RobustnessSpec | None = None

    def __post_init__(self) -> None:
        if self.headline and self.robustness is None:
            msg = f"headline claim {self.id!r} needs a robustness specification"
            raise ValueError(msg)


REGISTRY: dict[str, ClaimSpec] = {}


def claim(
    claim_id: str,
    *,
    title: str,
    question: str,
    method: str,
    headline: bool = False,
    robustness: RobustnessSpec | None = None,
) -> Callable[[ClaimFunction], ClaimFunction]:
    """Register ``function`` as the claim ``claim_id``."""

    def register(function: ClaimFunction) -> ClaimFunction:
        if claim_id in REGISTRY:
            msg = f"claim {claim_id!r} is registered twice"
            raise ValueError(msg)
        REGISTRY[claim_id] = ClaimSpec(
            id=claim_id,
            title=title,
            question=question,
            method=method,
            compute=function,
            headline=headline,
            robustness=robustness,
        )
        return function

    return register


def registered() -> dict[str, ClaimSpec]:
    """Every claim, after importing the modules that define them."""
    import strata.ledger.claims  # noqa: F401, PLC0415 - registration happens on import

    return dict(REGISTRY)
