"""The records of the evidence ledger.

A claim record is everything a reader needs to trust or doubt one number: what was asked, how it
was computed, on how many clients, with which uncertainty, under which caveats, whether it survives
stratification, and which data produced it (the fingerprint). Documents never hold numbers; they
hold references to these records.
"""

from __future__ import annotations

import enum
import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1

Verdict = Literal["holds", "weakens", "reverses", "insufficient"]
SourceKind = Literal["synthetic", "real"]


class Unit(enum.StrEnum):
    """How a value is rendered. A share is stored as a proportion and printed as a percentage."""

    SHARE = "share"
    COUNT = "count"
    AMOUNT = "amount"
    RATIO = "ratio"
    NUMBER = "number"
    DIFFERENCE = "difference"
    PVALUE = "p_value"
    DAYS = "days"


class Stratum(BaseModel):
    """The effect inside one stratum of one stratifier."""

    model_config = ConfigDict(frozen=True)

    stratum: str
    effect: float | None
    ci: tuple[float, float] | None = None
    n_treatment: int
    n_reference: int
    sufficient: bool
    suppressed: bool = False
    """True in the public view when the stratum holds too few clients to publish its numbers."""


class StratifierResult(BaseModel):
    """All strata of one stratifier, and the effect adjusted across them."""

    model_config = ConfigDict(frozen=True)

    stratifier: str
    strata: list[Stratum]
    adjusted_effect: float | None
    adjusted_ci: tuple[float, float] | None = None
    sign_flips: int
    reverses: bool


class Robustness(BaseModel):
    """The Simpson guard's result for a headline claim (ARCHITECTURE section 8)."""

    model_config = ConfigDict(frozen=True)

    group_col: str
    treatment: str
    reference: str
    metric: str
    overall_effect: float
    overall_ci: tuple[float, float]
    n_treatment: int
    n_reference: int
    stratifiers: list[StratifierResult]
    sign_flips: int
    shrinkage: float | None
    verdict: Verdict
    mechanism: str


class ClaimRecord(BaseModel):
    """One registered, reproducible claim."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    question: str
    method: str
    values: dict[str, float]
    units: dict[str, Unit]
    ci: dict[str, tuple[float, float]] = Field(default_factory=dict)
    interval_label: str = "95% CI"
    """What the intervals are: client-bootstrap confidence intervals by default, posterior
    intervals for Bayesian claims (``95% credible interval``, ``90% posterior interval``)."""
    value_n: dict[str, int] = Field(default_factory=dict)
    n: int
    n_events: int | None = None
    caveats: list[str] = Field(default_factory=list)
    meta: dict[str, str] = Field(default_factory=dict)
    headline: bool = False
    public: bool = False
    robustness: Robustness | None = None
    data_fingerprint: str
    source_kind: SourceKind
    computed_at: datetime
    strata_version: str

    @field_validator("values")
    @classmethod
    def _finite(cls, values: dict[str, float]) -> dict[str, float]:
        bad = [key for key, value in values.items() if not math.isfinite(value)]
        if bad:
            msg = f"non-finite values: {bad}"
            raise ValueError(msg)
        return values

    def value(self, key: str) -> float:
        if key not in self.values:
            msg = f"claim {self.id!r} has no value {key!r}"
            raise KeyError(msg)
        return self.values[key]


class Ledger(BaseModel):
    """A set of claims computed together from one warehouse."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    source_kind: SourceKind
    data_fingerprint: str
    computed_at: datetime
    claims: list[ClaimRecord]

    def get(self, claim_id: str) -> ClaimRecord:
        for record in self.claims:
            if record.id == claim_id:
                return record
        msg = f"no claim {claim_id!r} in this ledger"
        raise KeyError(msg)

    @property
    def ids(self) -> list[str]:
        return [record.id for record in self.claims]
