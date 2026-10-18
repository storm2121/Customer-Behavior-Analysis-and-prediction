# The robustness checker (a Simpson guard)

A headline finding compares a client-level metric between two groups: the share of clients who go
silent among branch-only and among digital-only clients, the share of branch-only clients among
older and younger clients. Such a comparison can be produced entirely by a third variable that
differs between the groups. The checker recomputes the comparison inside strata of each stratifier
and labels the headline **holds**, **weakens**, **reverses** or **insufficient**. The verdict is
stored in the claim record, shown as a badge in the documents and the dashboard, and listed for
every headline in the private report.

The rule was fixed before it was run on any data; it lives in `strata/ledger/robustness.py`.

## Definitions

For a metric $y$ (for example, $y = 1$ if a client is silent), a treatment group $T$ and a reference
group $R$:

- **Crude effect.** $\Delta = \bar y_T - \bar y_R$, the difference of means.
- **Stratum effect.** $\Delta_k = \bar y_{T,k} - \bar y_{R,k}$ inside stratum $k$ of a stratifier
  (activity tercile, age band, tenure band by default; a claim may choose its own).
- **Sufficient stratum.** Both groups have at least `min_group` clients in it (ten by default). Only
  sufficient strata enter the verdict; the others are listed by name, because a stratum where one
  group is absent is a lack of overlap (positivity), not evidence.
- **Adjusted effect.** The Mantel-Haenszel weighted mean of the sufficient stratum effects,

  $$\Delta_{\text{adj}} = \frac{\sum_k w_k \Delta_k}{\sum_k w_k}, \qquad w_k = \frac{n_{T,k}\, n_{R,k}}{n_{T,k} + n_{R,k}},$$

  which is the classical risk-difference estimator for stratified two-by-two tables.
- **Sign flips.** The number of sufficient strata whose effect has the opposite sign to $\Delta$.
- **Shrinkage.** $1 - \min_k \Delta_k / \Delta$ over sufficient strata (above one when a stratum
  flips sign).

## Verdict

1. **insufficient** when the crude effect is zero or no stratifier has a sufficient stratum;
2. otherwise **reverses** when, for at least one stratifier, $\Delta_{\text{adj}}$ has the opposite
   sign to $\Delta$;
3. otherwise **holds** when no sufficient stratum flips sign and shrinkage stays below one half;
4. otherwise **weakens**.

A reversal is a statement about direction, not significance: the interval of the adjusted effect is
reported next to it so a reader can see whether the reversed effect is distinguishable from zero.

## Uncertainty

Every effect carries a 95% percentile interval from the same client bootstrap as the rest of the
ledger: clients are resampled with replacement (multinomial weights), and the crude, stratum and
adjusted effects are recomputed on each resample.

## Choosing stratifiers

A stratifier should be a plausible common cause of group membership and outcome. Activity level is
the canonical one for silence, because a client with few events goes a quarter without any by
chance. It is deliberately left out for the age contrast: activity there is partly a consequence of
channel choice, and holding a consequence fixed biases the comparison.

## Publication

Robustness results of allow-listed claims are published with their verdict, crude and adjusted
effects and explanation. A stratum keeps its numbers only when each of its two groups holds at
least the publication floor of clients (`strata/ledger/allowlist.py`); otherwise only its label and
whether it was sufficient remain.
