"""Model-assembly tests that need no bank data."""

import pandas as pd

from customer_analytics import modeling


def test_one_hot_encoder_is_dense_on_any_sklearn_version():
    """`sparse` was renamed in 1.2 and removed in 1.4; both must work."""
    encoder = modeling._one_hot_encoder()
    encoded = encoder.fit_transform(pd.DataFrame({"colour": ["red", "blue", "red"]}))
    assert encoded.shape == (3, 2)
    assert hasattr(encoded, "tolist")  # dense array, not a sparse matrix


def test_drop_rare_classes_removes_classes_below_the_threshold():
    X = pd.DataFrame({"feature": range(10)})
    y = pd.Series([0] * 8 + [1, 1])
    X_kept, y_kept = modeling.drop_rare_classes(X, y, minimum=6)
    assert set(y_kept.unique()) == {0}
    assert len(X_kept) == 8


def test_drop_rare_classes_keeps_everything_when_all_are_common():
    X = pd.DataFrame({"feature": range(12)})
    y = pd.Series([0] * 6 + [1] * 6)
    X_kept, y_kept = modeling.drop_rare_classes(X, y, minimum=6)
    assert len(y_kept) == 12
