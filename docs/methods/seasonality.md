# Seasonality: the weekly cycle, the December wave and special weeks

Claims: `weekly_seasonality`, `december_deposit_wave`, `pre_eid_branch_deposits`,
`ramadan_card_effect`. Code: `strata/analysis/seasonality.py`.

## The weekly cycle (STL)

Each channel's daily event count $X_t$ (card, online, branch; every day) is decomposed by STL
(Cleveland, Cleveland, McRae and Terpenning, 1990) with a period of seven days and the robust
fitting option, which down-weights outlying days such as holidays:

$$X_t = T_t + S_t + R_t,$$

with trend $T$, weekly component $S$ and remainder $R$. The strength of each component
(Wang, Smith and Hyndman, 2006) is

$$F_S = \max\!\left(0,\; 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(S_t + R_t)}\right),
\qquad
F_T = \max\!\left(0,\; 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(T_t + R_t)}\right).$$

A strength near one means the component dominates the noise. Branch activity is weekly by
construction (branches close at weekends), so its weekly strength is a check on the data more than
a finding. A single year cannot carry a yearly STL period; yearly effects are read from the
calendar regression (`docs/methods/calendar.md`) and from the decompositions below.

Intervals come from the client bootstrap: the series is rebuilt from resampled clients,
$X^{(r)}_t = \sum_i w^{(r)}_i x_{it}$, and decomposed again.

## The December deposit wave: breadth and depth

For card deposits, let $n_{im}$ be client $i$'s number of deposits in month $m$, $a_{im}$ their
amount and $d_{im} = [n_{im} > 0]$ whether the client deposited at all. Over complete months (the
two-day stub at the end of the window is dropped) and with bootstrap weights $w_i$:

$$N_m = \sum_i w_i\, n_{im}, \qquad D_m = \sum_i w_i\, d_{im}, \qquad V_m = \frac{\sum_i w_i\, a_{im}}{D_m}.$$

With $O$ the other complete months, three ratios describe December:

- **count** $N_{\text{Dec}} / \bar N_O$: how much larger December's deposit activity is;
- **breadth** $D_{\text{Dec}} / \bar D_O$: how many more clients deposit at all;
- **depth** $V_{\text{Dec}} / \bar V_O$: how much more each depositor deposits.

A wave carried by breadth is many clients each making a seasonal deposit; one carried by depth is
the usual depositors depositing more. The count ratio also equals breadth times the ratio of
deposits per depositor.

## Special weeks against the same weekdays

Ramadan and the week before Eid al-Adha contain a particular mix of weekdays, and branch and card
activity differ strongly between weekdays. Comparing raw daily means would let the weekday mix
masquerade as an effect. With special days $\mathcal S$, baseline days $\mathcal B$ and weekday
$d$:

$$r_d = \frac{\bar Y_{\mathcal S, d}}{\bar Y_{\mathcal B, d}}, \qquad
r = \sum_d \frac{|\mathcal S_d|}{|\mathcal S|}\, r_d,$$

the ratio of daily means within each weekday, averaged with the special days' weekday mix as
weights. Weekdays without baseline days are skipped.

- `ramadan_card_effect`: Ramadan days against the other days of card coverage, Eid days excluded.
  A ratio close to one would be a finding too (a null result), so the claim is kept whatever its
  value.
- `pre_eid_branch_deposits`: the week before Eid al-Adha against other open weekdays (weekends,
  bank holidays and the Eids excluded), for counts and for amounts.

Intervals come from the client bootstrap, as everywhere in the ledger.
