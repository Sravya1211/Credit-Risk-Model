"""Tests for preprocessing, cost logic, and threshold selection."""

from __future__ import annotations

import numpy as np

from src import config
from src.data import get_splits
from src.features import build_preprocessor
from src.model import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    best_threshold,
    build_models,
    evaluate,
    expected_cost,
)


def test_preprocessor_produces_numeric_matrix() -> None:
    X_train, _, _, _ = get_splits()
    pre = build_preprocessor()
    transformed = pre.fit_transform(X_train)
    # One-hot encoding expands the 20 columns well past 20 numeric features.
    assert transformed.shape[0] == len(X_train)
    assert transformed.shape[1] > len(config.NUMERIC_FEATURES)
    assert np.isfinite(transformed).all()


def test_expected_cost_weights_false_negatives_more() -> None:
    y_true = np.array([1, 1, 0, 0])
    # One false negative (miss a default) vs one false positive.
    one_fn = np.array([0, 1, 0, 0])
    one_fp = np.array([1, 1, 1, 0])
    assert expected_cost(y_true, one_fn) == COST_FALSE_NEGATIVE
    assert expected_cost(y_true, one_fp) == COST_FALSE_POSITIVE
    assert expected_cost(y_true, one_fn) > expected_cost(y_true, one_fp)


def test_best_threshold_in_range() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_scores = rng.random(200)
    t = best_threshold(y_true, y_scores)
    assert 0.05 <= t <= 0.95


def test_models_train_and_rank_better_than_random() -> None:
    X_train, X_test, y_train, y_test = get_splits()
    model = build_models()["logistic_regression"]
    model.fit(X_train, y_train)
    result = evaluate("logistic_regression", model, X_test, y_test)
    # A useful model must beat a coin flip on ranking (ROC-AUC of 0.5).
    assert result.roc_auc > 0.65
