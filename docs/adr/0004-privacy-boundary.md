# ADR 0004: a structural privacy boundary with a public parametric generator and a private twin

Status: accepted (architect, 2026-10-18). Filled in M0; the twin is expanded in M9.
Refined by [ADR 0008](0008-private-zone-outside-the-working-tree.md).

## Context

The extract is an anonymised sample of a bank's clients provided for an academic challenge. The
repository is public and linked from CVs. The project must run end to end for any reader without the
data, and the real findings must stay private except for a few harmless aggregates.

A `.gitignore` entry is not a boundary: `git add -f`, a renamed file, a CSV dumped under another
extension, a notebook output or a log line pasted into a document all bypass it. The boundary has to
hold even when someone is careless.

## Decision

Real data is reachable only through `KASHF_DATA_DIR`; private outputs go to `KASHF_PRIVATE_DIR`; a
pre-commit guard blocks extract signatures, large CSVs and identifier-like tokens; an allowlist
decides which claims may be rendered publicly. The public demo uses a hand-specified parametric
generator that reproduces the schema and quirks without touching real rows. A separate private
generator fits a Gaussian copula on real marginals with Laplace noise and reports fidelity and
membership-inference risk; its output never enters the repository and is used only for local experiments.

The boundary is enforced in layers, each of which would stop a leak on its own:

1. **Paths.** `kashf.config.Settings` refuses a data or private folder that resolves inside the
   repository, and private writers check that their destination is under the private folder.
2. **Zones.** Commands run in an explicit public or private zone (ADR 0008).
3. **Commits.** `kashf privacy-check` blocks extract file names and headers, CSVs outside the two
   allowed folders or above the row limit, binary data formats, identifier-shaped tokens (shape
   learnt from the extract when it is available, a conservative pattern otherwise), ledger files with
   non-allow-listed claims or small cells, literal private locations and notebook outputs. It never
   echoes what it found.
4. **Publication.** Only allow-listed claims, each value computed on enough clients, reach
   `docs/ledger/claims.public.json`.
5. **Output.** Logs redact identifier fields and mask identifier-shaped tokens; the CLI never prints
   tracebacks with local variables.

## Alternatives considered

- **`.gitignore` alone.** Advisory only; rejected for the reasons above.
- **Encrypting the extract in the repository.** Still distributes the data, and the key becomes the
  problem.
- **Publishing only differentially private outputs.** Kept for the twin, which is a local tool; for
  documents, a short allowlist of coarse aggregates is easier to audit.

## Consequences

Two generators to maintain. The public demo's numbers are illustrative by construction, which the
docs state on every page that shows them. The private report is the artefact for interviews. Every
contributor runs the guard on every commit; false positives are resolved by changing the file,
never by bypassing the hook.
