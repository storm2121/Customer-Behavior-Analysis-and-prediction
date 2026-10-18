# A Markov chain of channel states and a what-if simulator

Claims: `channel_markov`, `markov_policy_simulation`. Code: `strata/analysis/markov.py`,
`strata/ledger/claims/markov.py`. `strata analyse markov` also writes the simulator's inputs and the
sample policy's bands to `analysis/markov_simulator.json` in the zone's workspace, for the API.

## States

Every client is in one state in every month of the window (`int_channel_state`, built on the dense
client-month panel):

| state | the month's events |
|---|---|
| `none` | no event in any channel |
| `branch_only` | branch operations, no card or online event |
| `digital_only` | card or online events, no branch operation |
| `mixed` | branch operations and card or online events |

A transition is a pair of consecutive months of one client. Transitions into the window's final
stub month (two days) are dropped, because a two-day month looks silent for almost everyone.
$N_{ij}$ counts the transitions from state $i$ to state $j$.

## Posterior of the transition matrix

Each row of the transition matrix $P$ is a probability vector. With a Dirichlet(1) prior (uniform
over the simplex) and multinomial counts, the posterior of row $i$ is conjugate and the rows are
independent:

$$P_{i\cdot} \mid N \sim \operatorname{Dirichlet}(1 + N_{i1}, \dots, 1 + N_{iK}).$$

Every derived quantity is computed on $S$ posterior draws $P^{(1)}, \dots, P^{(S)}$ (two thousand,
five hundred in quick runs), so its uncertainty follows from the counts with no further
approximation. Reported: posterior means of the transition probabilities and of the stationary
distribution, posterior medians of first-passage times, and 90% posterior intervals (5th to 95th
percentiles).

## Stationary distribution

The long-run mix $\pi$ solves $\pi P = \pi$ with $\sum_j \pi_j = 1$; numerically, the stacked
system $\begin{pmatrix} P^\top - I \\ \mathbf 1^\top \end{pmatrix} \pi = \begin{pmatrix} 0 \\ 1
\end{pmatrix}$ is solved by least squares. With every probability positive (the prior guarantees
it) the chain is irreducible and aperiodic, so $\pi$ is unique.

## Mean first-passage times

The expected number of months to first reach state $j$ from state $i$ comes from the fundamental
matrix (Kemeny and Snell, *Finite Markov Chains*, 1960):

$$Z = \left(I - P + \mathbf 1 \pi\right)^{-1}, \qquad
m_{ij} = \frac{z_{jj} - z_{ij}}{\pi_j} \quad (i \ne j), \qquad m_{jj} = \frac{1}{\pi_j}.$$

They satisfy $m_{ij} = 1 + \sum_{k \ne j} P_{ik}\, m_{kj}$: one month, then the remaining time from
wherever the month leads. The claim reports a few that answer business questions: how long until
a branch-only client first uses a digital channel, and until a digital client first has a silent
month.

## The simulator

A policy moves a share $p$ of the branch-only clients matching a filter (the sample policy: clients
under 45) to a target state (mixed use by default) at month zero. Starting from the state counts
$x_0$ of the latest complete month, with $E$ the eligible branch-only clients,

$$x^{\text{pol}}_0 = x_0 - pE\, e_{\text{branch\_only}} + pE\, e_{\text{target}},
\qquad x_{h+1} = x_h P^{(s)}, \quad h = 0, \dots, H - 1,$$

for every posterior draw $s$ and a horizon of $H = 12$ months. Three quantities are tracked:

- **branch load** $L_h = x_h \cdot \rho$, where $\rho_k$ is the mean number of branch operations
  per client-month in state $k$ (complete months);
- **digital share** $(x_{h,\text{digital\_only}} + x_{h,\text{mixed}}) / \sum_k x_{h,k}$;
- **silent share** $x_{h,\text{none}} / \sum_k x_{h,k}$.

The baseline is the same projection from $x_0$. Policy and baseline use the same draw $s$, so their
difference is taken draw by draw, and the claim reports the 5th, 50th and 95th percentiles of the
differences at months zero, one, three and twelve, together with the branch operations saved over
the horizon, $\sum_{h=1}^{H} (L^{\text{base}}_h - L^{\text{pol}}_h)$. The bands of each path are
what the API serves and the dashboard draws.

## Assumptions, and what they imply

- **First order and time-homogeneous.** Next month depends on this month only, through one matrix
  for the whole year. Seasonal moves (December, the summer) are averaged into it, and longer habits
  (a client who has used branches for years) are invisible to it.
- **Moved clients behave like their new state.** A moved client takes on the transitions and the
  branch rate of the target state. If mixed-use clients make more branch operations per month than
  branch-only clients (they are more active overall), a move to mixed use raises branch load in
  the model; the rates per state are reported with the claim so a reader can see why.
- **A one-off move fades.** The difference between policy and baseline decays like
  $|\lambda_2|^h$, with $\lambda_2$ the second-largest eigenvalue of $P$ in modulus, as both
  projections return to the stationary mix. The effect is largest in the first months.
- **Expected counts.** The projection propagates expected counts, so the bands carry the
  uncertainty of $P$, not the extra noise of a finite population or of the branch rates.
- **A closed population.** No client joins or leaves; `none` is a silent month, not a closed
  account.
