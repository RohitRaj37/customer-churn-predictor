import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from pathlib import Path


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    pos_label = 1 if y_test.dtype in ("int64", "int32", "float64") else "Yes"

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=pos_label),
        "recall": recall_score(y_test, y_pred, pos_label=pos_label),
        "f1": f1_score(y_test, y_pred, pos_label=pos_label),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")

    return metrics


def plot_confusion_matrix(model, X_test, y_test, ax=None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    y_pred = model.predict(X_test)
    labels = ["No", "Yes"]
    y_true = y_test.map({0: "No", 1: "Yes"}) if y_test.dtype in ("int64", "int32", "float64") else y_test
    y_pred_l = pd.Series(y_pred).map({0: "No", 1: "Yes"}) if y_pred.dtype in ("int64", "int32", "float64") else pd.Series(y_pred)
    cm = confusion_matrix(y_true, y_pred_l, labels=labels)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return ax.figure


def plot_roc_curve(model, X_test, y_test, ax=None, label: str = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    y_test_bin = y_test.map({"No": 0, "Yes": 1}) if y_test.dtype == "object" else y_test
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
    auc = roc_auc_score(y_test_bin, y_proba)
    label = label or f"ROC (AUC = {auc:.3f})"
    ax.plot(fpr, tpr, lw=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return ax.figure


def plot_pr_curve(model, X_test, y_test, ax=None, label: str = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    y_test_bin = y_test.map({"No": 0, "Yes": 1}) if y_test.dtype == "object" else y_test
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test_bin, y_proba)
    ap = average_precision_score(y_test_bin, y_proba)
    label = label or f"PR (AP = {ap:.3f})"
    ax.plot(recall, precision, lw=2, label=label)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    return ax.figure


def compare_models(
    models: dict, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    rows = []
    pos_label = 1 if y_test.dtype in ("int64", "int32", "float64") else "Yes"
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, pos_label=pos_label),
                "Recall": recall_score(y_test, y_pred, pos_label=pos_label),
                "F1": f1_score(y_test, y_pred, pos_label=pos_label),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
                "PR-AUC": average_precision_score(y_test, y_proba),
            }
        )
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).round(4)
