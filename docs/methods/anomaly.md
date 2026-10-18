# The monitoring layer: triage rules, detectors and a review queue

Code: `strata/models/anomaly/` (rules in `rules.yaml`), the review store in `strata/review/`.
Command: `strata anomaly run --budget <k>`, then `strata anomaly queue` and `strata anomaly
feedback <alert> <label>`.

The layer ranks clients for a person to look at. A rule that fires is a reason to look, never a
finding about anyone, and every alert carries the evidence a reviewer needs to dismiss it quickly.

## Rules

Configured in `strata/models/anomaly/rules.yaml`; the thresholds that dbt's `features_anomaly`
mart also uses are the project variables of `dbt/dbt_project.yml`, and a test keeps the two equal.

| rule | fires when | family |
|---|---|---|
| `large_cash_deposit` | a branch deposit reaches the reporting threshold | amount |
| `structuring` | several deposits, each below the threshold, together reach it within a window | splitting |
| `deposit_velocity` | many branch deposits within a window | frequency |
| `pass_through` | a large deposit is followed within days by a branch withdrawal of most of it | flow |
| `dormant_then_burst` | a busy month follows months without any event | activity |
| `cash_intensive` | branch deposits are large and far above the client's card spend | mix |

A client is a **pseudo-label** when rules of at least two different families fire: two independent
reasons to look.

## Detectors

**Isolation Forest** on the per-client features of `features_anomaly` (counts and amounts on a log
scale): the anomaly score is minus the forest's mean path-length score.

**Empirical-Bayes amount model.** For each category $c$ (channel and kind of event) and client $i$
with $n_i$ amounts, let $m_i$ be the mean log amount, $\mu$ the category's mean, $s^2$ the
within-client variance and $t^2$ the between-client variance of client means (net of the sampling
noise $s^2/n_i$). The client's typical log amount is shrunk toward the category's,

$$\hat m_i = \frac{t^2 m_i + (s^2/n_i)\,\mu}{t^2 + s^2/n_i},$$

so a client with few amounts is judged mostly against the category and one with many against
itself. Each amount gets a robust z,

$$z = \frac{\log a - \hat m_i}{1.4826 \cdot \operatorname{median}\,\lvert r - \operatorname{median}(r)\rvert},$$

with the residuals $r$ of the category, and a client's score is its largest $z$: a few amounts far
above the usual level stand out, a client who always moves large sums does not.

**Benford per client.** For clients with enough positive amounts, the mean absolute deviation of
their first digits from Benford's law (`docs/methods/benford.md`).

## The combined score and the queue

With the rule score $R_i$ (the weights of the rules that fire for client $i$ over the total weight)
and the percentile ranks $F_i$, $A_i$, $B_i$ of the three detectors (clients without enough amounts
for the Benford test get zero),

$$S_i = w_R R_i + w_F F_i + w_A A_i + w_B B_i,$$

with the weights of `rules.yaml`. Clients are ranked by $S_i$ (ties by client id) and the alert
budget keeps the top $k$: the queue reviewers can actually work through. Every alert lists its
reasons: each rule with its evidence (dates, amounts, counts), an unusual amount when its robust $z$
is at least three, a Benford nonconformity, an isolation in the forest's top percentile.

## Evaluation

- **Against pseudo-labels.** Precision at $k$ of the combined score, and of the detectors alone
  ($w_F F + w_A A + w_B B$). The first is partly circular, because the combined score contains the
  rules that define the pseudo-labels; the second says whether models that never saw the rules
  agree with them.
- **Against analysts.** Every alert of a run is stored with its rank; analysts label alerts
  `confirmed` (worth escalating, which is not guilt), `dismissed` or `needs_information`, and the
  precision at $k$ against feedback is the share of confirmed alerts among the labelled ones in the
  top $k$. The latest label of an alert counts.
- **On synthetic data**, the generator plants structuring and pass-through clients; a test checks
  that the queue finds most of them.

## The review store

Runs, alerts, feedback and annotations live in PostgreSQL (`STRATA_PG_URL`, the Compose service) or
a SQLite file in the workspace, created and evolved by Alembic migrations
(`strata/review/migrations`). Real alerts name real clients, so the private zone always uses a
SQLite file under `STRATA_PRIVATE_DIR` (ADR 0011).
