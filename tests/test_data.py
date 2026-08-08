"""Tests for src.data -- loading, target construction, and splitting."""

from __future__ import annotations

import pandas as pd

from src import config
from src.data import build_target, get_splits, load_raw


def test_load_raw_has_expected_shape() -> None:
    df = load_raw()
    assert df.shape == (1000, 21)
    assert config.RAW_TARGET in df.columns


def test_build_target_is_binary_and_inverted() -> None:
    raw = pd.DataFrame({config.RAW_TARGET: [1, 0, 1, 0]})
    out = build_target(raw)
    # good credit (1) -> not default (0); bad credit (0) -> default (1)
    assert out[config.TARGET].tolist() == [0, 1, 0, 1]
    assert config.RAW_TARGET not in out.columns


def test_splits_are_stratified_and_disjoint() -> None:
    X_train, X_test, y_train, y_test = get_splits()
    # No leakage: train and test indices do not overlap.
    assert set(X_train.index).isdisjoint(set(X_test.index))
    # Stratification keeps the default rate close in both splits.
    assert abs(y_train.mean() - y_test.mean()) < 0.05
    # Every feature column is present, target is not.
    expected = set(config.NUMERIC_FEATURES) | set(config.CATEGORICAL_FEATURES)
    assert set(X_train.columns) == expected
