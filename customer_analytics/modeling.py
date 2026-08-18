"""Encoding, class balancing, and the Random Forest transaction-type model."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from . import config

logger = logging.getLogger(__name__)


def _one_hot_encoder() -> OneHotEncoder:
    """Build a dense OneHotEncoder across sklearn versions.

    The parameter was renamed `sparse` -> `sparse_output` in scikit-learn 1.2
    and `sparse` was removed in 1.4, so pinning either name breaks half the
    installed base.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass
class Dataset:
    """Model-ready matrices plus the encoders needed to interpret them."""

    X: pd.DataFrame
    y: pd.Series
    label_encoder: LabelEncoder
    encoder: OneHotEncoder


def build_features(merged: pd.DataFrame) -> Dataset:
    """One-hot the categoricals, label-encode the target, drop rare classes."""
    merged = merged.copy()

    for col in config.CATEGORICAL_FEATURES:
        if col not in merged.columns:
            merged[col] = "Unknown"
        merged[col] = merged[col].fillna("Unknown")

    encoder = _one_hot_encoder()
    encoded = pd.DataFrame(
        encoder.fit_transform(merged[config.CATEGORICAL_FEATURES]),
        columns=encoder.get_feature_names_out(config.CATEGORICAL_FEATURES),
        index=merged.index,
    )

    numerical = [col for col in config.NUMERICAL_FEATURES if col in merged.columns]
    X = pd.concat([merged[numerical], encoded], axis=1)

    target = merged[config.TARGET].astype(str).fillna("Unknown")
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(target), index=merged.index)

    X, y = drop_rare_classes(X, y)
    return Dataset(X=X, y=y, label_encoder=label_encoder, encoder=encoder)


def drop_rare_classes(X: pd.DataFrame, y: pd.Series, minimum: int | None = None):
    """Remove classes too small for SMOTE to synthesise neighbours from."""
    minimum = minimum or config.MIN_CLASS_SAMPLES
    counts = y.value_counts()
    rare = counts[counts < minimum].index

    if len(rare) == 0:
        return X, y

    logger.warning("Dropping %d class(es) with fewer than %d samples", len(rare), minimum)
    keep = ~y.isin(rare)
    return X[keep], y[keep]


def balance(X: pd.DataFrame, y: pd.Series):
    """Oversample minority classes so the forest is not majority-dominated."""
    smote = SMOTE(random_state=config.RANDOM_STATE)
    return smote.fit_resample(X, y)


@dataclass
class TrainedModel:
    """A fitted forest with the split it was evaluated on."""

    model: RandomForestClassifier
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    accuracy: float
    report: str

    def feature_importances(self, top: int = 10) -> pd.DataFrame:
        """Most influential features, highest first."""
        return (
            pd.DataFrame(
                {"Feature": self.X_train.columns, "Importance": self.model.feature_importances_}
            )
            .sort_values("Importance", ascending=False)
            .head(top)
            .reset_index(drop=True)
        )


def train(dataset: Dataset) -> TrainedModel:
    """Balance, split, fit and score the classifier."""
    X_resampled, y_resampled = balance(dataset.X, dataset.y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return TrainedModel(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        accuracy=accuracy_score(y_test, predictions),
        report=classification_report(y_test, predictions, zero_division=0),
    )


def sample_predictions(trained: TrainedModel, dataset: Dataset, n: int = 10) -> pd.DataFrame:
    """Side-by-side actual vs predicted labels for a handful of test rows."""
    samples = trained.X_test.sample(n, random_state=config.RANDOM_STATE)
    actual = trained.y_test.loc[samples.index]
    predicted = trained.model.predict(samples)

    return pd.DataFrame(
        {
            "Actual": dataset.label_encoder.inverse_transform(actual),
            "Predicted": dataset.label_encoder.inverse_transform(predicted),
        }
    )
