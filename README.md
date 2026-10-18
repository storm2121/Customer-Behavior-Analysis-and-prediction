# Strata

**Strata** (كشف: an account statement, and the act of uncovering) is a retail-banking behaviour
analytics system built on a rolling-year, anonymised extract of card transactions, online
operations and branch operations of a sample of clients, provided by CIH Bank for an academic
challenge. It replaces a hackathon script from 2024 whose bugs are documented openly in
[`legacy/README.md`](legacy/README.md).

> **Status: under reconstruction.** The rebuild follows [`ARCHITECTURE.md`](ARCHITECTURE.md) and
> [`MILESTONES.md`](MILESTONES.md); progress is logged in [`PROGRESS.md`](PROGRESS.md).

## What makes it different

- **No number is typed by hand.** Every figure in the documentation is rendered from an evidence
  ledger of registered, bootstrapped claims, and CI fails when a document drifts from the ledger or
  contains a number that does not come from it.
- **Findings are stress-tested.** A robustness checker re-computes each headline finding within
  strata of activity, age and tenure and labels it *holds*, *weakens* or *reverses*.
- **The privacy boundary is structural.** The real extract is reachable only through
  `STRATA_DATA_DIR`, private outputs only go to `STRATA_PRIVATE_DIR`, and a pre-commit guard blocks
  anything shaped like the extract. The public repository runs end to end on synthetic data from a
  hand-specified generator. See [`PRIVACY.md`](PRIVACY.md).
- **Time is respected.** Client-level outcomes are validated forward in time, never with random
  splits.

<!-- ledger:start -->
## What the data says

Rendered from the evidence ledger; see [the findings](docs/findings/index.md) for the tables,
methods and caveats behind each sentence.

- **Channels.** 51% of clients use both the branch and a
  digital channel, 29% only digital channels and
  15% only the branch
  (n = 1,000 clients; 2023-10-02 to 2024-10-02; real extract, allow-listed aggregate; data fingerprint 85887a44d747).
- **Age.** The branch-only share is
  6.9% in the <25 band and
  33% in the 65+ band, while the
  gap between the two gender codes is -1.5 percentage points.
  Robustness verdict: **holds**
  (n = 1,000 clients; 2023-10-02 to 2024-10-02; real extract, allow-listed aggregate; data fingerprint 85887a44d747).
- **The week.** On cards, the busiest day of the week carries
  1.10× the events of the quietest
  (n = 947 clients, 54,852 events; 2023-10-02 to 2024-10-02; real extract, allow-listed aggregate; data fingerprint 85887a44d747).
- **Prices.** Card purchase amounts show acceptable conformity with Benford's
  law; the share of leading nines differs from the law's expectation by
  +3.8 percentage points,
  the usual mark of psychological pricing (n = 515 clients, 14,496 events; 2023-10-02 to 2024-10-02; real extract, allow-listed aggregate; data fingerprint 85887a44d747).
<!-- ledger:end -->

## Quick start (synthetic data, no access to the extract needed)

```bash
uv sync                 # Python 3.12 environment from uv.lock
uv run pre-commit install
just check              # format, lint, types, tests with coverage, privacy scan
uv run strata --help
```

[`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/) are the only tools to install
by hand (`uv tool install rust-just` provides `just`).

## Repository layout

| Path | Contents |
|---|---|
| `strata/` | The Python package: configuration, CLI, contracts, ingest, synthetic data, ledger, analysis, models, reports, privacy guard. |
| `tests/` | `pytest` and `hypothesis` suites; every test runs on synthetic data. |
| `docs/adr/` | Architecture decision records. |
| `legacy/` | The 2024 script and the 2025 package, frozen, with a list of their bugs. |

## Licence and data

The code is published for reading and review. The bank extract is confidential and is not, and will
never be, part of this repository.
