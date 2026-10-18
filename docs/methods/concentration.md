# Concentration: Lorenz curves, Gini coefficients, top shares and RFM scores

Claims: `spend_concentration`, `rfm_by_channel`. Code: `strata/analysis/concentration.py`.

Every function takes optional weights $w_i$, the number of times client $i$ was drawn in a
bootstrap resample, so a resample is evaluated without materialising repeated rows. Without weights,
$w_i = 1$.

## Lorenz curve and Gini coefficient

Sort clients by value $x_{(1)} \le \dots \le x_{(n)}$ (with their weights). The Lorenz curve joins
the points

$$\left(\frac{\sum_{j \le k} w_{(j)}}{\sum_j w_j},\; \frac{\sum_{j \le k} w_{(j)} x_{(j)}}{\sum_j w_j x_j}\right),
\qquad k = 0, \dots, n,$$

the share of value held by the smallest share of clients. The Gini coefficient is one minus twice
the area under that curve, computed exactly by the trapezoid rule on its points:

$$G = 1 - \sum_{k=1}^{n} (P_k - P_{k-1})(L_k + L_{k-1}),$$

with $P$ and $L$ the population and value coordinates. $G = 0$ when every client holds the same
value and approaches one when a single client holds it all.

## Top shares

The share of the total held by the top fraction $q$ of clients: clients are taken from the largest
value down until their weights reach $q \sum_j w_j$, and the client who crosses that line counts with
the fraction of their weight that fits. Partial counting keeps the statistic smooth across
resamples, where whole-client counting would jump.

## RFM scores per channel

For the clients active in a channel (at least one event in it), from `client_360`:

| channel | recency | frequency | monetary value |
|---|---|---|---|
| card | days from the last card event to the window end | card events | card spend |
| online | days from the last online operation | online operations | online amounts |
| branch | days from the last branch operation | branch operations | branch credits |

Each measure becomes a score from one to five by quintile of rank. Ranks are average ranks, so tied
values share a score, and with $r_i$ the rank of client $i$ among $n$ (from one),

$$s_i = 1 + \left\lfloor \frac{5\,(r_i - 1)}{n} \right\rfloor .$$

For recency the order is reversed: a recent client scores high. The claim reports, per channel,
the median recency, frequency and monetary value, the share of the channel's value held by its best
clients (scores of at least four on all three measures) and the Gini coefficient of frequency.

Intervals are client-bootstrap percentiles. The quintile cut-offs, and so the set of best clients,
are held at their full-sample values across resamples; the intervals therefore describe the value
those clients hold, not the uncertainty of who they are.
