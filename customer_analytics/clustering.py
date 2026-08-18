"""KMeans customer segmentation with PCA projection for visualisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config


def _kmeans(n_clusters: int) -> KMeans:
    """KMeans with an explicit n_init.

    The default changed to 'auto' in scikit-learn 1.4 and warns loudly before
    that, so it is set explicitly rather than left to the version in use.
    """
    return KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=10)


@dataclass
class Segmentation:
    """Cluster assignments plus everything needed to explain them."""

    labels: np.ndarray
    profiles: pd.DataFrame
    components: pd.DataFrame
    scaled: np.ndarray


def scale(merged: pd.DataFrame) -> np.ndarray:
    """Standardise the clustering features; KMeans is distance-based."""
    features = [col for col in config.CLUSTERING_FEATURES if col in merged.columns]
    X = merged[features].fillna(merged[features].mean())
    return StandardScaler().fit_transform(X)


def elbow_scores(scaled: np.ndarray, max_k: int = 10) -> list[float]:
    """Within-cluster sum of squares for k = 1..max_k."""
    return [_kmeans(k).fit(scaled).inertia_ for k in range(1, max_k + 1)]


def segment(merged: pd.DataFrame, n_clusters: int | None = None) -> Segmentation:
    """Assign clusters and project to two components for plotting."""
    n_clusters = n_clusters or config.N_CLUSTERS
    scaled = scale(merged)

    labels = _kmeans(n_clusters).fit_predict(scaled)

    features = [col for col in config.CLUSTERING_FEATURES if col in merged.columns]
    profiles = merged.assign(cluster=labels).groupby("cluster")[features].mean()

    components = pd.DataFrame(
        PCA(n_components=2).fit_transform(scaled), columns=["PC1", "PC2"]
    )
    components["cluster"] = labels

    return Segmentation(labels=labels, profiles=profiles, components=components, scaled=scaled)
