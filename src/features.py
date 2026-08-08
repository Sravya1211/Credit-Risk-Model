"""Preprocessing: scale numeric columns and one-hot encode categorical ones.

Everything is wrapped in a single scikit-learn ColumnTransformer so the exact
same transformation is applied at train time and at prediction time. That
consistency is what prevents "train/serve skew" -- a subtle bug where a model
behaves differently in production than in the notebook.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Return the preprocessing transformer for the credit features.

    - Numeric features are standardized (mean 0, unit variance) so that the
      logistic-regression coefficients are comparable to one another.
    - Categorical features are one-hot encoded. `handle_unknown="ignore"` means
      a category never seen during training will not crash prediction -- it is
      simply encoded as all-zeros, which is the safe behavior for a live model.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
