# ADR 0013: The private twin is a differentially private copula of the client table

Status: accepted (architect, 2026-10-19). Written in M9, revised in M11. Expands the twin of
[ADR 0004](0004-privacy-boundary.md) and ARCHITECTURE section 5.3.

## Context

ADR 0004 keeps a second generator for local experiments: a copy of the real data's structure with
differential privacy, whose fidelity and membership-inference risk are measured before any use.
ARCHITECTURE section 5.3 sketches it as a Gaussian copula over real marginals with Laplace noise on
each marginal histogram, sampled at a thousand clients, reporting per-column Kolmogorov-Smirnov
distances, a correlation-matrix distance, a propensity-score AUC, a nearest-neighbour distance ratio
and a membership-inference AUC, with any output path inside the repository refused.

Two points of that sketch needed a decision. A copula learns its correlation matrix from the real
rows too, so noise on the marginals alone would not make the twin differentially private. And the
five extract files are event logs keyed by client; a copula of events would break the clients apart,
while the behaviour the project studies lives at client level.

## Decision

`strata --private synth-twin` (module `strata.synthetic.twin`) builds the twin of
`marts.client_360` in the private warehouse, the table of one row per client that the warehouse
derives from `STRATA_DATA_DIR`. By default it reproduces a reduced set of client columns (age,
gender code, channel profile, active months and the event counts of the three channels); with
`--columns full` it adds tenure, the marital code, the total of events, the amounts by channel and
the branch share. Identifiers and dates are never among them.

- **Privacy accounting.** Neighbouring tables differ in one client's row; the number of clients is
  public (the public claims publish it). The budget `--epsilon` (default one) is split evenly between
  marginals and correlations, and each half evenly between its columns or pairs.
- **Marginals.** Histograms over bins fixed in advance from public bounds and domains, never read
  from the data: banded ages and tenures, geometric bins for counts and amounts, tenths for the
  share, the public code lists for categories, with catch-all bins for unknown and missing values.
  Point masses (no amount, a share of exactly zero or one) get bins of almost no width so the twin
  keeps them. Values are clipped to the bounds; Laplace noise of scale two over the column's budget
  is added to every count.
- **Dependence.** Kendall's tau-a for every pair of columns, which one client's row moves by at most
  four over the number of clients less one, with Laplace noise scaled to all pairs at once, mapped
  to a correlation by the sine rule and projected onto the correlation matrices.
- **Mechanism.** Laplace noise (pure epsilon-DP) by default. `--mechanism gaussian` calibrates
  Gaussian noise with the analytic Gaussian mechanism to the L2 sensitivity of all histograms
  together and of the whole vector of taus, for (epsilon, delta)-DP, with `--delta` (default one in
  a million) split evenly between the two halves.
- **Sampling.** Correlated normals through the noisy marginals; post-processing, which costs no
  privacy. A thousand rows by default (`--rows`).
- **Metrics** (`metrics.json` beside `twin.csv`). The twin is fitted on a random part of the clients;
  the rest is a holdout. Fidelity: Kolmogorov-Smirnov or total-variation distance per column, the
  mean and largest absolute difference of the Spearman matrices, and the cross-validated AUC of a
  logistic regression telling twin rows from real ones. Privacy: distances from twin rows and from
  holdout rows to their closest training row (medians, their ratio, and the share of twin rows
  closer than the holdout's fifth percentile), the share of binned twin rows equal to a binned
  training row against the holdout's share, and a distance-based membership-inference attack (AUC
  and advantage on training members against holdout non-members).
- **Refusals, before anything is read.** In CI; any destination inside the repository, whatever the
  zone; any destination outside `STRATA_PRIVATE_DIR`; the public zone. Tests assert each one. The
  twin carries no synthetic marker, so a public ingest refuses it like any unmarked extract, and the
  command prints file names only, never a metric.

## What the private runs showed (in words; the values stay in the private folder)

A first sweep of the budget on the reference instance, which holds about a thousand clients, kept
the membership-inference attack at chance and no twin row closer to a training row than holdout
rows are, at every budget tried. Fidelity was the price: with all fourteen columns, at a budget of
one most marginals were visibly distorted and the correlations mostly noise, because the columns
share the budget and ninety-one pairs share the correlation half of it.

A second, time-boxed experiment (M11) compared, at the same budgets and over five seeds each, the
Laplace and Gaussian mechanisms on the full and the reduced column set, judged on the reduced
set's columns so that the four variants are comparable. At a budget of one, the reduced set with
Laplace noise distorted clearly fewer marginals than the full set, a real-or-twin classifier no
longer told it from the data, and it never raised a privacy flag; the Gaussian mechanism improved
the full set somewhat and the reduced set a little more, but on the reduced set it once let the
membership-inference attack rise above chance. The reduced Laplace twin is therefore the default:
better fidelity with privacy still at chance. Its correlations remain noisy at a budget of one; at
a budget of two its marginals and correlations are close. The parametric generator stays the source
of every public demo, and the twin remains a local tool whose metrics must be read before any use.

## Alternatives considered

- **Noise on the marginals only, as first sketched.** The correlation matrix would then be an exact
  function of the real rows; rejected.
- **A twin of the five event files.** Keeps the file shapes, but a copula cannot keep a client's
  events together, and the client-level table is what the analyses use.
- **The Gaussian mechanism as the default.** Its noise grows with the square root of the number of
  pairs rather than with the number, and it helped the full column set; on the reduced set it added
  little and once raised the membership flag, and it needs a delta. Kept as an option.
- **All fourteen columns by default.** More of the client table, but at the default budget the
  noise spread over fourteen marginals and ninety-one pairs leaves too little structure. Kept as an
  option (`--columns full`) for larger budgets.
- **Learned generators (CTGAN, PATE-GAN, private-PGM).** Better fidelity on large tables; heavy
  dependencies and harder accounting for a thousand clients.

## Consequences

The twin is honest about its trade-off: its metrics say how far it is from the data and how little it
reveals about any one client, and on a sample of this size they say that differential privacy at a
small budget leaves little structure. Using the twin for anything public would need its own ADR.
