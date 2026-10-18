# Seeds

Code maps for every categorical value in the extracts. They are the single source of truth for
vocabularies: dbt staging joins them, and the Python package (`kashf.contracts.vocab`) reads them
for the contracts, the data-quality checks and the synthetic generator.

The `evidence` and `semantics` columns say why a meaning is believed: `label` (evident from the
label text), `observed` (supported by the data, for example clients coded "no" have no online
operations) or `assumed` (not documented by the bank; treat with care). Nothing here is a value
taken from a client row.
