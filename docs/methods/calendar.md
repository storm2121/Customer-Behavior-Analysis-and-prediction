# Calendar effects: a negative-binomial regression and its Bayesian twin

Claims: `calendar_effects` (frequentist), `calendar_bayes` (Bayesian). Code:
`strata/analysis/calendar.py`, `strata/analysis/calendar_bayes.py`. Design decisions: ADR 0009
(terms), ADR 0007 (backend).

## Series

Every series is a daily count from the `daily_counts` mart, which is dense: a day without events is
a zero, not a missing row, because a count model needs the zeros. Days outside a file's coverage
(the card file ends a few days before the others) are dropped.

| key | counts | days modelled |
|---|---|---|
| `card_all` | every card event | every day |
| `card_withdrawal` | card withdrawals | every day |
| `card_deposit` | card deposits | every day |
| `branch_deposit` | branch deposit operations | open days: Monday to Friday outside bank holidays |
| `online_all` | every online operation | every day |

## Frequentist model

For day $t$ of a series, $Y_t$ follows a negative binomial distribution with mean $\mu_t$ and the
NB2 variance $\operatorname{Var}(Y_t) = \mu_t + \alpha\,\mu_t^2$, with a log link:

$$
\log \mu_t = \beta_0
+ \sum_{d \ne \text{Mon}} \gamma_d\,[\text{dow}_t = d]
+ \sum_{b \ne \text{11–15}} \delta_b\,[\text{bin}_t = b]
+ \sum_{m < M} \theta_m\, c_{m,t}
+ \sum_{p} \lambda_p\,[t \in p].
$$

- **Day of week** $d$, Monday as reference.
- **Day-of-month bins** $b$ (days one to five, six to ten, and so on), days eleven to fifteen as
  reference, so $e^{\delta_{1–5}}$ is the salary-day effect.
- **Month** in sum-to-zero coding over the $M$ months of the window,
  $c_{m,t} = [\text{month}_t = m] - [\text{month}_t = M]$, which gives $\theta_M = -\sum_{m<M}
  \theta_m$: each $e^{\theta_m}$ is month $m$ against the unweighted average month. December's
  effect is $e^{\theta_{\text{Dec}}}$. There is no separate trend term (ADR 0009): over one year
  the month terms absorb it.
- **Periods** $p$: Ramadan, Eid al-Fitr, Eid al-Adha, the week before Eid al-Adha (dates from
  `dbt/seeds/calendar_periods.csv`) and the other bank holidays, national and religious
  (`dbt/seeds/bank_holidays.csv`). The Eids are bank holidays too, but have their own terms, so
  every period is read against ordinary days. The Moroccan dates follow the official moon sighting
  and are tagged `assumed` in the seeds until confirmed.

A column that is constant on the modelled days (Saturday on an open-days series, say) is dropped.
Every effect is reported as a rate ratio $e^{\text{coefficient}}$: the factor by which the expected
daily count changes.

### Estimation

$\beta$ and $\alpha$ are estimated by maximum likelihood (statsmodels, NB2). When $\hat\alpha$ sits
at the boundary the series is close to Poisson; $\hat\alpha$ is floored at $10^{-8}$ so the family
stays valid. Rate ratios are computed on a log scale bounded at $\pm 30$, because a rare indicator
can have an enormous standard error.

### Intervals: a client bootstrap

Days are not independent units: the same clients produce every day, and a salaried client withdraws
at the start of every month. The intervals therefore resample clients, like the rest of the ledger
(`docs/methods/robustness.md`). Let $y_{it}$ be client $i$'s count on day $t$ and $w^{(r)}$ the
multinomial weights of resample $r$ (how many times each client is drawn). The resampled series is

$$Y^{(r)}_t = \sum_i w^{(r)}_i\, y_{it},$$

refitted by iteratively reweighted least squares at the fitted dispersion $\hat\alpha$; the 2.5th
and 97.5th percentiles of the refitted rate ratios form the interval. Full runs use two hundred
refits per series, quick runs thirty. Holding $\alpha$ fixed makes each refit a GLM fit rather
than a full likelihood maximisation, at the cost of ignoring the dispersion's own uncertainty; the
claim says so. Wald intervals from the model fit are kept in the code for comparison.

## Bayesian model

The same likelihood and design, with priors:

$$
\beta_0 \sim \mathcal N(\log \bar y,\, 1), \qquad
\beta_j \sim \mathcal N(0,\, 0.5^2), \qquad
r \sim \operatorname{Exponential}(0.1),
$$

where PyMC parameterises the negative binomial by $r$ with $\operatorname{Var}(Y) = \mu + \mu^2 / r$
(so $\alpha = 1/r$ above, and the prior mean of $r$ is ten). $\mathcal N(0, 0.5^2)$ on a log rate
ratio puts 95% of the prior on rate ratios between about $0.38$ and $2.7$: weak for the terms with
many days, where the data dominate, and deliberately shrinking for terms with very few days (the
two-day Eids).

### Posterior summaries

For each rate ratio, the posterior median and the equal-tailed 95% credible interval. December's
effect is computed draw by draw from the sum-coded month coefficients, which gives

$$P\big(e^{\theta_{\text{Dec}}} > 2 \mid y\big) = \frac{1}{S}\sum_{s=1}^{S} \big[e^{\theta^{(s)}_{\text{Dec}}} > 2\big]$$

over the $S$ posterior draws. With a single December in the data this probability cannot separate a
seasonal wave from a one-off event in the same weeks; the claim carries that caveat.

### Checks

- **Posterior predictive coverage.** Replicated series $y^{\text{rep}}$ are drawn from the
  posterior predictive distribution. The coverage is the share of days whose observed count lies in
  the central 90% predictive interval; a well-calibrated model gives about nine days in ten.
- **A test quantity.** $T(y) = \operatorname{sd}(y)$, with the posterior predictive p-value
  $P\big(T(y^{\text{rep}}) \ge T(y) \mid y\big)$. A value near zero or one means the model misses the
  spread of the series (for example, extra variability the calendar terms do not explain).
- **Convergence.** The largest rank-normalised split $\hat R$ (it should be below 1.01), the
  smallest bulk effective sample size and the number of divergent transitions, all stored with the
  claim.

The sampler is NUTS; full runs use four chains of a thousand draws after a thousand tuning steps,
quick runs two chains of three hundred after three hundred (ADR 0007).

## Reading the two together

The frequentist fit gives point estimates with client-bootstrap intervals; the Bayesian fit gives
posterior medians, credible intervals and probabilities of statements. Where the data are plentiful
the two agree closely; where they differ, the prior is doing the work, which is the signal that a
term rests on very few days.
