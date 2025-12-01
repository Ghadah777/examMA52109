###
## Tests for preprocessing.py
## MA52109 - Task 3
###

import numpy as np
import pandas as pd
import pytest

from cluster_maker.preprocessing import select_features, standardise_features


# -------------------------------------------------------------
# Test 1:
# This test checks that select_features correctly detects missing
# columns. If this fails, the clustering pipeline may run with
# wrong or incomplete features, producing incorrect results.
# -------------------------------------------------------------
def test_select_features_missing_column():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    with pytest.raises(KeyError):
        select_features(df, ["a", "c"])   # "c" does not exist


# -------------------------------------------------------------
# Test 2:
# This test ensures select_features rejects non-numeric columns.
# If non-numeric data is silently accepted, the clustering pipeline
# will fail later or produce meaningless results.
# -------------------------------------------------------------
def test_select_features_non_numeric():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": ["a", "b", "c"]  # non-numeric
    })

    with pytest.raises(TypeError):
        select_features(df, ["x", "y"])


# -------------------------------------------------------------
# Test 3:
# This test checks that standardise_features really standardises
# the data so that each feature has mean ~0 and std ~1.
# Incorrect behaviour here would distort clustering distances.
# -------------------------------------------------------------
def test_standardise_features_mean_std():
    X = np.array([[1.0, 10.0],
                  [2.0, 20.0],
                  [3.0, 30.0]])

    X_scaled = standardise_features(X)

    # Means should be close to 0
    assert np.allclose(X_scaled.mean(axis=0), [0.0, 0.0], atol=1e-7)

    # Standard deviations should be close to 1
    assert np.allclose(X_scaled.std(axis=0), [1.0, 1.0], atol=1e-7)
