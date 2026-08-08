"""End-to-end training run.

Run it with:  python -m src.train

It loads the data, trains both models, evaluates them under the bank's cost
matrix, writes a metrics report to reports/metrics.json, and saves every
figure referenced in the README to reports/figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so plots save without a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from . import config
from .data import get_splits
from .model import build_models, evaluate, expected_cost


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _plot_threshold_curve(y_true, y_scores, chosen: float, out_path) -> None:
    """Show how total business cost changes as the decision threshold moves."""
    thresholds = np.linspace(0.05, 0.95, 181)
    costs = [expected_cost(y_true, (y_scores >= t).astype(int)) for t in thresholds]
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, costs, label="total cost")
    plt.axvline(chosen, color="red", linestyle="--", label=f"chosen = {chosen:.2f}")
    plt.axvline(0.5, color="grey", linestyle=":", label="naive = 0.50")
    plt.xlabel("decision threshold")
    plt.ylabel("total business cost (test set)")
    plt.title("Cost-optimal threshold beats the naive 0.50 cut-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> dict:
    """Train, evaluate, and persist all reporting artifacts. Returns metrics."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = get_splits()
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} "
          f"| Default rate: {y_train.mean():.1%}")

    models = build_models()
    evaluations = {}
    fitted = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        evaluations[name] = evaluate(name, pipeline, X_test, y_test)
        e = evaluations[name]
        print(
            f"\n[{name}]"
            f"\n  ROC-AUC : {e.roc_auc:.3f}"
            f"\n  PR-AUC  : {e.pr_auc:.3f}"
            f"\n  cost @0.50 threshold  : {e.cost_at_half:.0f}"
            f"\n  cost @{e.threshold:.2f} threshold : {e.cost_at_threshold:.0f}"
        )

    # --- Figures ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipeline in fitted.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey")
    ax.set_title("ROC curve: Logistic Regression vs Random Forest")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_comparison.png", dpi=150)
    plt.close(fig)

    # Pick the model with the lower business cost as the deployed model.
    best_name = min(evaluations, key=lambda n: evaluations[n].cost_at_threshold)
    best_pipeline = fitted[best_name]
    best_eval = evaluations[best_name]
    print(f"\nSelected model (lowest cost): {best_name}")

    y_scores = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= best_eval.threshold).astype(int)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["good", "default"], cmap="Blues"
    )
    disp.ax_.set_title(f"{best_name} @ threshold {best_eval.threshold:.2f}")
    disp.figure_.tight_layout()
    disp.figure_.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(disp.figure_)

    _plot_threshold_curve(
        np.asarray(y_test), y_scores, best_eval.threshold,
        FIGURES_DIR / "threshold_cost.png",
    )

    # --- Persist metrics ----------------------------------------------------
    report = {
        "dataset": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "default_rate": round(float(y_train.mean()), 3),
        },
        "cost_matrix": {
            "false_negative": 5.0,
            "false_positive": 1.0,
        },
        "models": {n: e.as_dict() for n, e in evaluations.items()},
        "selected_model": best_name,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote reports/metrics.json and 3 figures to reports/figures/")
    return report


if __name__ == "__main__":
    main()
