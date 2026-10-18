# ADR 0010: the canonical raw layer, its landing tables and its provenance

Status: accepted (executing agent, M4.5, 2026-10-18). Extends ARCHITECTURE.md section 5.0.

## Context

ARCHITECTURE.md section 5.0 makes the pipeline source-agnostic: a mapping file per source loads it
into `raw.clients`, `raw.events` (client_id, ts, channel, kind, amount, direction, origin,
source_file, source_row) and the optional `raw.client_aggregates`, and dbt staging reads only those
tables. Milestone M4.5 adds the layer "thin, over the existing contracts", and its definition of
done asks that the reference instance keep its manifest, its data-quality report and its marts.

Three details are not settled by the architecture:

1. The reference extract's branch file carries a currency (`devise`), and the exact-repeat
   definition of branch operations includes it. The canonical event columns have no place for it.
2. The data-quality report of M1 is written against the reference files' own columns (the
   aggregate file's empty rows, the branch labels' assumed directions, and so on). A canonical
   table cannot answer those questions.
3. The data fingerprint hashes the manifest. If the manifest records the mapping, the fingerprint
   of the same bytes changes, and so does every claim's provenance; if it does not, two mappings
   of the same bytes could be merged in one ledger.

## Decision

- **A `currency` column** joins the canonical event columns. It is optional, like `direction` and
  `origin`; the reference mapping fills it from `devise` and the branch staging model's repeat
  definition uses it, as before.
- **Landing tables.** A file read through one of the reference extract's typed contracts is also
  written, exactly as parsed, to `landing.<extract>`. The source-specific checks of `strata dq` run
  on the landing tables; a source without such a contract gets the source-agnostic checks instead
  (exact repeats, events outside the window, zero amounts, events without a client), computed on
  the canonical tables. Exact repeats are reported under the reference extract's name of each
  channel, so dbt's reconciliation of the repeat flags holds for every source. dbt never reads the
  landing tables.
- **Provenance.** The manifest records the mapping's name and digest, but the fingerprint keeps
  hashing the files only: it names the bytes. The ledger records the mapping's digest next to the
  fingerprint, and a partial build merges only into a ledger of the same bytes read through the
  same mapping.
- **Code maps.** A mapping translates a source's channel, kind and direction codes into the
  reference vocabulary of the seeds (`kind_codes`, `channel_codes`, `direction_codes`). Every value
  must translate, or the ingest stops; the seeds stay the single definition of what a kind means.

## Consequences

- The reference instance is reproduced exactly: on the synthetic extract (a test against a
  snapshot taken before mappings existed) and on the real one (compared privately), the manifest's
  files and fingerprint, the data-quality report and every model's row count are unchanged, and so
  is every claim.
- A third, generated schema (other names, delimiter, decimal mark, date format, encoding, codes,
  row order and file layout) gives the same canonical tables and the same marts (a property test).
- A warehouse built before this change needs `strata dbt build --full-refresh` once: staging now
  reads tables that did not exist.
