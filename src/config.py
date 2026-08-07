"""Central configuration: paths and column groups.

Keeping every setting in one place means the rest of the code never hard-codes a
column name or a magic number -- which makes the pipeline easy to audit and reuse.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_FILE = DATA_DIR / "german_credit.csv"

# Public mirror of the UCI German Credit dataset; cached locally after first run.
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"

# --- Columns ----------------------------------------------------------------
RAW_TARGET = "credit_risk"   # raw label in the file: 1 = good loan, 0 = bad loan
TARGET = "default"           # our modeling target: 1 = default (bad), 0 = good

NUMERIC_FEATURES = [
    "duration", "amount", "installment_rate",
    "present_residence", "age", "number_credits", "people_liable",
]

CATEGORICAL_FEATURES = [
    "status", "credit_history", "purpose", "savings",
    "employment_duration", "personal_status_sex", "other_debtors",
    "property", "other_installment_plans", "housing",
    "job", "telephone", "foreign_worker",
]

# --- Reproducibility & split ------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.25
