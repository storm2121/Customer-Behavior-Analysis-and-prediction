"""Paths, column names and tunables.

Everything the original script hardcoded inline lives here, so switching data
directory or retuning the model never means editing analysis code.
"""

from __future__ import annotations

import os
from pathlib import Path

# Override with COMPANION_DATA_DIR=/path/to/csvs; defaults to ./data
DATA_DIR = Path(os.environ.get("CIH_DATA_DIR", "data"))

CSV_FILES = {
    "clients": "echantillons_clients(in).csv",
    "transactions": "transactions_monetiques_annee_glissante_anonymise(in).csv",
    "online_operations": "operations_en_ligne_annee_glissante_anonymise(in).csv",
    "agency_operations": "operations_en_agence_annee_glissante_anonymise(in).csv",
    "al_akhawayn": "al_akhawayn_dataset_v2(in).csv",
}

CSV_ENCODING = "latin1"

# The source extracts are large; the original analysis worked from a sample.
SAMPLE_SIZE = 1000
RANDOM_STATE = 42

CLIENT_ID = "identifiant_client"
TARGET = "libelle_nature_transaction"

NUMERICAL_FEATURES = [
    "age",
    "anciennete_client",
    "montant_local",
    "montant_transaction",
    "montant_operation",
    "days_since_last_transaction",
    "avg_transaction_amount",
]

CATEGORICAL_FEATURES = ["genre", "situation_familiale", "service", "operation"]

CLUSTERING_FEATURES = NUMERICAL_FEATURES

# Columns that must survive the merge, or the run is not worth continuing.
REQUIRED_COLUMNS = [
    "age",
    "anciennete_client",
    "genre",
    "situation_familiale",
    "montant_local",
    TARGET,
]

AGE_BINS = [18, 30, 45, 60, 75, 100]
AGE_LABELS = ["18-30", "31-45", "46-60", "61-75", "76+"]

TEST_SIZE = 0.3
N_ESTIMATORS = 100
# SMOTE needs enough neighbours; classes below this are dropped before resampling.
MIN_CLASS_SAMPLES = 6
N_CLUSTERS = 3
