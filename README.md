# Credit Risk Model — Loan Default Prediction

## Problem
Given a loan application, predict whether the loan will **default** (go unpaid)
before the money is lent. This is a **supervised binary classification** problem:
- **Instances:** 1,000 past loans
- **Features:** 20 application attributes (loan amount, age, credit history, etc.)
- **Target:** `default` — 1 if the loan went bad, 0 if it was repaid

## Dataset
UCI German Credit dataset (1,000 real loans). The classes are **imbalanced**
(~30% defaults), and features carry clear signal — e.g., applicants with under
100 DM in their checking account default ~49% of the time vs. ~12% for those
who bank elsewhere. This makes the problem both learnable and realistic.

## Methodology

### 1. Train / test split
The data is split **75% training / 25% test**, stratified on the target so the
~30% default rate is preserved in both halves. The test set is held out and used
only for final evaluation — this is how we measure whether the model
**generalizes** to new applicants rather than memorizing the ones it saw.

### 2. Preventing data leakage
Any information that wouldn't be available at loan-application time is a
**leakage** risk. Two concrete steps guard against it:
- The raw `credit_risk` label is dropped the moment the `default` target is
  built, so the model can never read the answer directly.
- All preprocessing is fit on training data only (see Step 3).

A fixed `random_state=42` makes the split — and every result that flows from it
— fully **reproducible**.


### 3. Preprocessing
Two column groups get two different treatments, bundled into one
`ColumnTransformer` so both are applied identically at train and prediction time:

- **Numeric features** are **standardized** (`StandardScaler`) so a large-range
  column like `amount` doesn't dominate a small-range one like `age`.
- **Categorical features** are **one-hot encoded** (`OneHotEncoder`) so no
  fake ordering is invented among unordered categories like `housing`.
  `handle_unknown="ignore"` keeps the model from crashing when a new category
  appears in production.

Wrapping preprocessing in a scikit-learn `Pipeline` guarantees fit statistics
(means, category lists) are learned from the **training set only** — closing
off preprocessing as a second source of data leakage.

### 4. Models
Two models are trained side by side, on purpose:

- **Logistic Regression** — the industry standard for credit scoring for 60
  years. Its coefficients are directly interpretable, which is why regulators
  are comfortable with it.
- **Random Forest** — an ensemble of decision trees that captures non-linear
  patterns the logistic model can't. Used as a benchmark: if it doesn't beat
  the simple model by much, the interpretable choice wins by default.

Both use `class_weight="balanced"` to handle the ~30/70 class imbalance so the
optimizer doesn't ignore the minority (defaulting) class.

A custom **cost function** replaces the usual accuracy metric: a missed default
is charged 5× as much as a needless rejection, matching the German Credit
dataset's official cost matrix. This is the business objective we optimize
against in Step 5.


### 5. Evaluation
Two ranking metrics are reported, both **threshold-independent** (they judge
how well the model separates defaults from non-defaults across all cutoffs):

- **ROC-AUC**: probability that a random defaulter gets a higher risk score
  than a random non-defaulter. Range 0.5 (random) → 1.0 (perfect).
- **PR-AUC**: the same idea using precision-recall, which is more informative
  when the positive class is rare (as with our ~30% default rate).

Both are computed alongside the business cost, which is what the model is
ultimately optimized for.

## How to run

    git clone https://github.com/Sravya1211/Credit-Risk-Model.git
    cd Credit-Risk-Model
    pip install -r requirements.txt
    python -m src.train

    ### 6. Explainability (SHAP)
Under the US Equal Credit Opportunity Act, a lender that denies credit must
send the applicant an **adverse action notice** listing the specific principal
reasons. A black-box model can't produce that document, so it can't be
deployed. This project uses **SHAP** (SHapley Additive exPlanations) to
generate the reason codes automatically.

- **Global explanations** rank the features driving default risk across the
  whole population. Used for model auditing and fair-lending review.
- **Local explanations** attribute a single applicant's risk score to the
  features that raised or lowered it. This is the adverse-action document.

The example reason codes for one test applicant are saved to
`reports/metrics.json` under `example_adverse_action_reasons`.
