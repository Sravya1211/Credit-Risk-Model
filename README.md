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
