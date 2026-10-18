# ADR 0003: time-based validation for every client-level outcome

Status: accepted (architect, 2026-10-18). Filled in M0; expanded with results during M5.

## Context

The legacy model balanced classes with SMOTE before a random split, so synthetic neighbours of test
rows were in the training set and the reported accuracy was inflated. Client outcomes such as silence
are temporal; a random split lets the model see the future.

The extract covers a single rolling year, so there is little room: every month spent on the outcome
window is a month taken from the features.

## Decision

Observation window 2023-10-02 to 2024-06-30 for features, outcome window 2024-07-01 to 2024-09-29
for labels, with rolling-origin backtests at 2024-04-30, 2024-05-31 and 2024-06-30. Resampling, if
used at all, happens inside the training fold only. A test asserts no feature reads past its cutoff.

The cutoffs and their outcome windows live in a dbt seed, so the SQL that builds features and the
Python that evaluates models read the same definition.

## Implementation (M5)

- `strata.models.splits` reads the cutoffs from the `silence_cutoffs` seed, as dbt does. Each
  cutoff is one problem: features from `marts.features_silence` (events on or before the cutoff),
  the label from the three months after it.
- Backtests train on one cutoff and test on the next, so a model is always evaluated on a later
  origin than the one it learnt from; the primary backtest tests on the primary cutoff. LightGBM's
  hyperparameters are tuned by Optuna with cross-validation inside the earliest cutoff only;
  calibration (isotonic and Platt) is fitted on a held-out fifth of the training clients.
- With one year of data, the outcome windows of consecutive cutoffs overlap by two months, so a
  backtest's training labels share calendar weeks with its test labels. This is the price of three
  origins in one year; every model card states it.
- The leakage test rewrites every event after the earliest cutoff (drops half, triples the
  amounts of the rest, duplicates them), rebuilds the features with dbt, and requires the earliest
  cutoff's features to be unchanged while the later cutoffs' features do change.
- The adoption models have no time axis (the aggregate file is one cross-section): they are
  evaluated out of fold over stratified folds, and the online operation counts that define their
  targets are never features.

## Alternatives considered

- **Random k-fold on clients.** Rejected: it mixes calendar periods, so seasonal behaviour in the
  outcome window leaks into training.
- **Grouped k-fold by client without a time cut.** Prevents a client appearing on both sides but
  still trains on the future.
- **A single holdout cutoff.** Simpler, but one cutoff cannot show whether a model is stable; three
  rolling origins can.

## Consequences

Fewer training examples and wider intervals, reported honestly with bootstrap CIs. Results are
comparable across cutoffs, which exposes instability instead of hiding it.
