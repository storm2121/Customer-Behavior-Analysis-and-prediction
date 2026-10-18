# CIH Bank — Customer Behaviour Analysis & Prediction

Analyses retail banking customer behaviour to support digital-payment adoption in
Morocco: it merges five anonymised extracts into a per-client view, engineers
behavioural features, predicts transaction type with a Random Forest, and segments
customers with KMeans.

Built as a data-analytics project at Al Akhawayn University.

---

## Quickstart

```bash
pip install -r requirements.txt
python cli.py train
```

| Command | What it does |
|---|---|
| `python cli.py explore` | Dataset summary and the four exploratory plots |
| `python cli.py train` | Fits the classifier, prints accuracy, report and feature importances |
| `python cli.py cluster` | KMeans segmentation with elbow curve and PCA projection |
| `python cli.py gui` | Trains, then opens the Tkinter prediction window |

Useful flags: `--data-dir PATH`, `--sample-size N`, `--clusters K`, `--no-plots`
(compute without rendering figures, for headless runs).

## Data

**The datasets are confidential and are not distributed with this repository.**
`.gitignore` excludes `data/` and all CSVs so they cannot be committed by accident.

Place the five extracts in `data/` (or point `--data-dir` elsewhere):

```
data/
├── echantillons_clients(in).csv
├── transactions_monetiques_annee_glissante_anonymise(in).csv
├── operations_en_ligne_annee_glissante_anonymise(in).csv
├── operations_en_agence_annee_glissante_anonymise(in).csv
└── al_akhawayn_dataset_v2(in).csv
```

Missing files are reported by name at startup rather than failing deep inside pandas.

## Structure

```
customer_analytics/
├── config.py          paths, column names, model hyperparameters
├── data.py            loading, client sampling, aggregation, merging, imputation
├── features.py        derived features — pure functions, fully unit tested
├── modeling.py        encoding, SMOTE balancing, Random Forest, evaluation
├── clustering.py      scaling, elbow scores, KMeans, PCA projection
├── visualization.py   plots, each with an explicit show flag
└── gui.py             Tkinter front-end
cli.py                 command-line entry point
tests/                 pytest suite, runs without the datasets
```

## Pipeline

1. **Load & sample** — read the five extracts, sample 1,000 clients, and restrict
   the satellite tables to those ids before aggregating.
2. **Aggregate** — collapse transactions, online and agency operations to one row
   per client (sum of amounts, modal category, latest date).
3. **Merge** — left-join onto the client table; Al Akhawayn demographics backfill
   missing age and marital status.
4. **Engineer** — `days_since_last_transaction`, `avg_transaction_amount`, `age_group`.
5. **Model** — one-hot encode, drop classes too small for SMOTE, oversample, then
   fit a 100-tree Random Forest on a 70/30 split.
6. **Segment** — standardise, choose k via the elbow method, cluster, project to 2 PCs.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers feature engineering and model assembly, and needs no datasets —
`features.py` is deliberately free of file access and global state.

## Notes on the model

- **SMOTE** balances the target so the forest is not dominated by the majority
  transaction type. Classes with fewer than 6 samples are dropped first, since
  SMOTE cannot synthesise neighbours from fewer.
- **Sampling** is fixed at `random_state=42` throughout, so runs are reproducible.
- One-hot encoding is built to work across scikit-learn versions — the relevant
  parameter was renamed in 1.2 and removed in 1.4.
