"""What each kind of model interval is, in words, for the documents and the API that show one.

Kept apart from the modules that compute the intervals so that a reader of the labels (the ledger,
the API) does not import scikit-learn to name a percentile range.
"""

from __future__ import annotations

METRIC_INTERVAL = "95% client-bootstrap interval"
"""Every model metric: the 2.5th to 97.5th percentiles over resamples of the test clients."""

STABILITY_INTERVAL = "95% range of refits"
"""Segment stability: the 2.5th to 97.5th percentiles of the refits' adjusted Rand index."""
