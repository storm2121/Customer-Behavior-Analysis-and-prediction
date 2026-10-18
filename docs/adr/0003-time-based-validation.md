# ADR 0003: time-based validation for every client-level outcome

Status: accepted (architect, 2026-10-18). Expand during M5.

## Context

The legacy model balanced classes with SMOTE before a random split, so synthetic neighbours of test
rows were in the training set and the reported accuracy was inflated. Client outcomes such as silence
are temporal; a random split lets the model see the future.

## Decision

Observation window 2023-10-02 to 2024-06-30 for features, outcome window 2024-07-01 to 2024-09-29
for labels, with rolling-origin backtests at 2024-04-30, 2024-05-31 and 2024-06-30. Resampling, if
used at all, happens inside the training fold only. A test asserts no feature reads past its cutoff.

## Consequences

Fewer training examples and wider intervals, reported honestly with bootstrap CIs. Results are
comparable across cutoffs, which exposes instability instead of hiding it.
