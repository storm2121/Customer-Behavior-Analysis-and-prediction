# ADR 0007: PyMC for the Bayesian calendar model

Status: accepted (executing agent, M4, 2026-10-18). Records the choice that ARCHITECTURE.md section 9
leaves to this ADR.

## Context

ARCHITECTURE.md section 9 asks for the calendar regression in a Bayesian form, with weakly
informative priors, posterior predictive checks and posterior probabilities such as
P(December deposit effect > 2×). It fixes an install order: PyMC on Python 3.12; if PyMC does not
install cleanly within thirty minutes, NumPyro; if that fails too, a Laplace approximation of the
GLM.

The reference machine runs Windows 11 without a C++ compiler. PyMC compiles its models through
PyTensor, which uses a C backend when `g++` is present and otherwise falls back to a slower
Python and NumPy backend, warning about it on every import.

## Decision

- **Backend:** PyMC 6 with PyTensor 3 and ArviZ 1 (versions pinned in `uv.lock`). They installed
  from wheels through `uv sync` in a few minutes, well inside the thirty-minute budget, so neither
  fallback was needed and neither is implemented.
- **No compiler assumed:** `strata.analysis.calendar_bayes` sets `PYTENSOR_FLAGS=cxx=` (unless the
  variable is already set) before importing PyMC. PyTensor then uses its Python backend
  deliberately rather than after a failed probe, and the warning disappears. A machine with a
  compiler can set `PYTENSOR_FLAGS` itself to use the C backend; the results do not depend on it.
- **Sampler:** NUTS, chains run one after the other in a single process (`cores=1`), seeded with
  the master seed, so a rerun on the same data and versions draws the same posterior. Two effort
  levels: full (four chains of a thousand draws after a thousand tuning steps) for real runs, and
  quick (two chains of three hundred after three hundred) for tests, CI and demos. The ledger
  records which one produced a claim (`meta.sampler`, and `meta.effort` for quick builds), and
  `--public` refuses quick builds.
- **Model:** the frequentist design matrix of `strata.analysis.calendar` (ADR 0009) with the
  priors of `docs/methods/calendar.md`; the same series and terms, so the two fits can be read side
  by side.

## Consequences

- Sampling the two series of the `calendar_bayes` claim takes seconds to minutes on the reference
  machine, not hours; `strata analyse calendar` stays interactive.
- Convergence is checked on every run, not assumed: the claim stores the largest R-hat, the
  smallest bulk effective sample size and the number of divergent transitions, and the methods page
  says how to read them.
- Switching backend later (NumPyro for speed, or a Laplace approximation) means implementing
  `fit_bayes` again behind the same `BayesFit` result; nothing else changes.
