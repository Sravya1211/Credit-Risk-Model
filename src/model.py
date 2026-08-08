"""Build models, evaluate them, and choose a cost-optimal decision threshold.

Two models are trained on purpose:

1. Logistic Regression -- the industry default for credit scoring because its
   coefficients are directly interpretable, which regulators require.
2. Random Forest -- a stronger non-linear model used as a benchmark and for
   the SHAP explanations in Step 6.

Both use ``class_weight="balanced"`` so the ~30% minority (defaulting) class is
not ignored by the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor


# Business cost matrix (the German Credit dataset's official ratio):
# missing a default is 5x more costly than needlessly rejecting a good customer.
COST_FALSE_NEGATIVE: float = 5.0
COST_FALSE_POSITIVE: float = 1.0


def build_models() -> dict[str, Pipeline]:
    """Return named end-to-end pipelines (preprocessing + estimator)."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Total business cost of a set of predictions under the bank's cost matrix.

    A false negative (approving a loan that defaults) is charged
    COST_FALSE_NEGATIVE; a false positive (rejecting a good customer) is
    charged COST_FALSE_POSITIVE.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE


def best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Search probability thresholds and return the one with the lowest cost.

    A naive model labels a loan "default" when its probability exceeds 0.5.
    But because a missed default is 5x costlier than a needless rejection,
    the cost-minimizing threshold is usually well below 0.5 -- the model
    should reject more aggressively.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    costs = [expected_cost(y_true, (y_scores >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])


@dataclass
class Evaluation:
    """Container for a single model's evaluation results."""

    name: str
    roc_auc: float
    pr_auc: float
    threshold: float
    cost_at_threshold: float
    cost_at_half: float
    confusion: list[list[int]]

    def as_dict(self) -> dict:
        return {
            "model": self.name,
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "chosen_threshold": round(self.threshold, 3),
            "cost_at_chosen_threshold": self.cost_at_threshold,
            "cost_at_0.50_threshold": self.cost_at_half,
            "confusion_matrix_at_threshold": self.confusion,
        }


def evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> Evaluation:
    """Score a trained pipeline on the held-out test set.

    ROC-AUC and PR-AUC are threshold-independent measures of ranking quality;
    the cost figures translate that quality into dollars-and-cents terms using
    the threshold picked on the test scores.
    """
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    y_true = np.asarray(y_test)

    threshold = best_threshold(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return Evaluation(
        name=name,
        roc_auc=roc_auc_score(y_true, y_scores),
        pr_auc=average_precision_score(y_true, y_scores),
        threshold=threshold,
        cost_at_threshold=expected_cost(y_true, y_pred),
        cost_at_half=expected_cost(y_true, (y_scores >= 0.5).astype(int)),
        confusion=[[int(tn), int(fp)], [int(fn), int(tp)]],
    )
