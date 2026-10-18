# Legacy code (frozen)

This directory keeps the code that Strata replaces, unchanged, so that every criticism below can be
checked against the source. Nothing here is imported, tested, linted or type-checked by the new
build, and nothing here should be refactored: it is evidence, not a dependency.

| Path | What it is |
|---|---|
| `2024_script/Analysis.py` | The original hackathon script, restored verbatim from the first upload in the git history. |
| `2025_package/` | The 2025 restructuring of that script into a package (`customer_analytics/`), its CLI, its tests, its requirements and its README. |

The 2025 package fixed several crashes and usability bugs of the script, but kept its analytical
design, so most of the bugs below apply to both. They are stated here in plain words because the
results they produced were reported as if they were sound.

## Bugs that invalidate the results

1. **Oversampling before the split.** SMOTE is applied to the whole dataset and the train/test split
   is drawn from the resampled rows (`Analysis.py`; `modeling.train` in the package). Synthetic
   neighbours of test rows are therefore in the training set, and the reported accuracy is inflated
   by construction. Resampling, if used at all, belongs inside the training fold.
2. **A join that matched nothing, silently.** The aggregate file (`al_akhawayn_dataset_v2`) is a
   different sample of clients: its identifiers match none of the client file. The left join on
   `identifier` therefore matched zero rows, so the advertised "backfill of age and marital status
   from the aggregate file" did nothing, and every column taken from that file was empty. No check
   counted the matches.
3. **Recency measured from the day the script runs.** `days_since_last_transaction` subtracts the last
   card date from `pd.to_datetime('today')` instead of from the end of the extract. The feature grows
   by one every day the script is re-run, and its scale depends on when it was run. The package made
   `today` injectable for tests, but production calls still use the run date.
4. **A weak target.** The model predicts each client's most frequent card transaction type, a label
   dominated by one category and computed from the same transactions whose sums are used as
   features. It answers no decision a bank would take.
5. **A spurious class from missing labels.** Clients with no card transactions have a missing target;
   `astype(str)` runs before `fillna('Unknown')`, so the missing value becomes the literal string
   `'nan'` and is learned as a class of its own.
6. **No time-based validation and no uncertainty.** Client-level outcomes are evaluated with a random
   split; no metric carries a confidence interval or a baseline.

## Bugs that mislead the reader

7. **Misattributed sample output (script).** The final "predictions vs actual" table looks up client
   identifiers and ages in `merged_data` by the positional index of rows in the resampled test set.
   Those rows are mostly synthetic and their index has no relation to clients, so the identifiers and
   ages printed next to each prediction belong to unrelated clients. Printing identifiers at all was
   also a privacy problem; the package dropped that column.
8. **Mislabelled charts.** "Frequency of Digital Payments by Age Group" and "by Gender" count clients
   (one row per client), not payments.
9. **A misnamed feature.** `avg_transaction_amount` is total card spend divided by tenure in years: card
   spend per year of tenure, computed from a single year of transactions. It is not an average
   transaction amount.
10. **Absence imputed as average.** Missing channel amounts (clients who never used the branch or the
    online channel) are filled with the column mean, so "no activity" is indistinguishable from
    "average activity".
11. **Rare classes dropped without saying so in the results.** Classes too small for SMOTE are removed
    before training, which silently changes the question being answered.
12. **Exact repeat rows counted as distinct transactions.** Card rows that repeat client, date, amount,
    category and origin cannot be told apart from double postings (there is no transaction id and no
    time of day), and nothing flags them.
13. **All warnings suppressed (script).** `warnings.filterwarnings('ignore')` hides every pandas and
    scikit-learn warning, including the ones that would have pointed at the problems above.
14. **A sample that samples nothing.** The client file has as many rows as the requested sample size,
    so the "sample" is a reshuffle of the whole file.

## Crashes and usability bugs fixed by the 2025 package

- `OneHotEncoder(sparse=False)` fails on scikit-learn 1.4 and later (the argument was renamed).
- The "random prediction" button passed a fixed `random_state`, so it always showed the same client.
- The manual-input window was defined but never attached to a button, so it was unreachable.

## What replaces this

See `ARCHITECTURE.md` at the repository root: a typed warehouse with data-quality checks that count
the repeat rows and the empty join, time-based validation with bootstrap intervals, and an evidence
ledger that keeps every published number traceable to the code that produced it.
