"""Load the German Credit data and split it into train / test sets.

Downloads the file once, caches it locally, builds the target, and produces a
stratified train/test split.
"""

from __future__ import annotations

import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def download_data(force: bool = False) -> None:
    """Download the raw dataset to the local cache if it isn't already there."""
    if config.DATA_FILE.exists() and not force:
        return
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(config.DATA_URL, config.DATA_FILE)


def load_raw() -> pd.DataFrame:
    """Return the raw dataset as a DataFrame, downloading it first if needed."""
    download_data()
    return pd.read_csv(config.DATA_FILE)


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary `default` column (1 = bad loan) and drop the raw label.

    Dropping the original label is a deliberate anti-leakage step: if it stayed
    among the features, the model could read the answer directly.
    """
    out = df.copy()
    out[config.TARGET] = (out[config.RAW_TARGET] == 0).astype(int)
    out = out.drop(columns=[config.RAW_TARGET])
    return out


def get_splits():
    """Return a stratified `X_train, X_test, y_train, y_test` split.

    Stratifying on the target keeps the ~30% default rate identical in both
    splits, so our test score isn't distorted by an unlucky shuffle.
    """
    df = build_target(load_raw())
    feature_cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[config.TARGET]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
