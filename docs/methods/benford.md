# Leading digits: Benford's law with Nigrini's conformity bands

Claims: `benford_purchases` (public), `benford_by_category`, `benford_two_digits`. Code:
`strata/analysis/benford.py`.

## The law

Amounts that arise from many multiplicative processes spread evenly on a logarithmic scale, and
their leading digits then follow Benford's law. For the first significant digit $D_1$ and the first
two significant digits $D_1D_2$:

$$P(D_1 = d) = \log_{10}\!\left(1 + \frac{1}{d}\right), \quad d = 1, \dots, 9; \qquad
P(D_1D_2 = k) = \log_{10}\!\left(1 + \frac{1}{k}\right), \quad k = 10, \dots, 99.$$

Digits are read from the scientific notation of each positive amount, which is exact: a formula
based on $\lfloor \log_{10} x \rfloor$ misreads amounts such as $0.3$ through floating-point drift.
Zero and negative amounts have no leading digit and are left out; exact repeated rows are excluded,
as everywhere in the warehouse.

## Tests

With observed shares $\hat p_k$ of $n$ amounts over $K$ cells (nine or ninety) and the expected
shares $p_k$:

- **Mean absolute deviation**, $\text{MAD} = \frac{1}{K}\sum_k |\hat p_k - p_k|$, read against
  Nigrini's conformity bands (*Benford's Law*, 2012):

  | test | close | acceptable | marginally acceptable | nonconformity |
  |---|---|---|---|---|
  | first digit | up to 0.006 | up to 0.012 | up to 0.015 | above 0.015 |
  | first two digits | up to 0.0012 | up to 0.0018 | up to 0.0022 | above 0.0022 |

- **Chi-square**, $\chi^2 = n \sum_k (\hat p_k - p_k)^2 / p_k$ on $K - 1$ degrees of freedom. With
  many thousands of amounts it rejects for deviations far too small to matter, which is why the
  MAD band is the recommended reading and the p-value is reported only for completeness.

## Categories

`benford_by_category` runs the first-digit test on every category with at least a hundred amounts
(card purchases, e-commerce, withdrawals and deposits; online bill payments; branch deposits) and
the first-two-digit test where there are at least a thousand. Some categories are not expected to
conform at all: withdrawals are multiples of banknotes, and set prices cluster below round numbers.
Their nonconformity describes how the amounts are made, not anything suspicious.

## Psychological pricing

Prices just below a round number (a price ending in nine, ninety-nine or nine hundred) put more
amounts on a leading nine than the law predicts. `benford_purchases` reports the excess
$\hat p_9 - p_9$, and `benford_two_digits` the three leading pairs most over-represented against
the law, which is where pricing habits show.

## Intervals

Each client contributes a row of digit counts $C_{i\cdot}$; a bootstrap resample with weights $w$
gives shares $\hat p^{(r)} = w^{(r)\top} C \,/\, w^{(r)\top} C \mathbf 1$, from which the MAD, the
excess of nines and the excess of each pair are recomputed. Resampling clients rather than amounts
keeps a client's habits (a monthly transfer of the same amount, say) together.
