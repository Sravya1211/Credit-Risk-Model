"""Model explainability with SHAP.

Why this file exists: under the US Equal Credit Opportunity Act, a lender that
denies credit must give the applicant specific reasons. A model that cannot
explain an individual decision is therefore not deployable, no matter how
accurate it is. SHAP attributes each prediction to the features that drove it,
which is exactly the "reason codes" a compliance team needs.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so plots save without a display
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.pipeline import Pipeline


def _feature_names(pipeline: Pipeline) -> np.ndarray:
    """Recover human-readable names after one-hot encoding."""
    return pipeline.named_steps["preprocess"].get_feature_names_out()


def global_importance_plot(pipeline: Pipeline, X_sample, out_path) -> None:
    """Save a SHAP summary plot ranking which features drive default risk overall.

    X_sample should be a modest slice of the training data (a few hundred rows)
    -- SHAP is exact for trees but still benefits from a bounded sample.
    """
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["clf"]

    X_transformed = preprocess.transform(X_sample)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # Newer SHAP returns a (n_samples, n_features, n_classes) array for RF;
    # select the positive (default) class.
    values = shap_values[..., 1] if np.ndim(shap_values) == 3 else shap_values

    plt.figure()
    shap.summary_plot(
        values,
        X_transformed,
        feature_names=_feature_names(pipeline),
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def top_reasons_for_applicant(
    pipeline: Pipeline, X_row, n: int = 5
) -> list[tuple[str, float]]:
    """Return the top ``n`` features pushing one applicant toward default.

    This is the machine-readable version of an "adverse action reason code":
    each tuple is a feature name and its signed contribution to the risk score.
    """
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["clf"]

    X_transformed = preprocess.transform(X_row)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    values = shap_values[..., 1] if np.ndim(shap_values) == 3 else shap_values
    contributions = values[0]

    names = _feature_names(pipeline)
    order = np.argsort(np.abs(contributions))[::-1][:n]
    return [(str(names[i]), float(contributions[i])) for i in order]
