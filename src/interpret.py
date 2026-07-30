import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path


def _extract_shap(shap_values, class_idx: int = 1):
    if isinstance(shap_values, list):
        return shap_values[class_idx]
    if shap_values.ndim == 3:
        return shap_values[:, :, class_idx]
    return shap_values


def _extract_expected_value(ev, class_idx: int = 1):
    if isinstance(ev, list):
        return ev[class_idx]
    if isinstance(ev, np.ndarray) and ev.ndim > 0:
        return ev[class_idx]
    return ev


def explain_with_shap(model, X: pd.DataFrame, sample_size: int = 50) -> shap.Explainer:
    if "XGB" in type(model).__name__ or "RandomForest" in type(model).__name__:
        explainer = shap.TreeExplainer(model)
    else:
        bg = shap.sample(X, min(sample_size, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)
    return explainer


def plot_shap_summary(
    model, X: pd.DataFrame, max_display: int = 15, sample_size: int = 100
) -> plt.Figure:
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
    explainer = explain_with_shap(model, X_sample, sample_size=sample_size)
    shap_values = _extract_shap(explainer.shap_values(X_sample))
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.title("SHAP Summary Plot", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_shap_waterfall(
    model, X: pd.DataFrame, idx: int = 0, max_display: int = 10
) -> plt.Figure:
    explainer = explain_with_shap(model, X)
    raw_sv = explainer.shap_values(X.iloc[[idx]])
    shap_vals = _extract_shap(raw_sv)[0]
    expected_value = _extract_expected_value(explainer.expected_value)

    fig = plt.figure(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals,
            base_values=expected_value,
            data=X.iloc[idx].values,
            feature_names=X.columns.tolist(),
        ),
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Waterfall for Customer #{idx}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def get_top_factors(
    model, X_row: pd.DataFrame, feature_names: list[str], top_n: int = 5
) -> list[dict]:
    explainer = explain_with_shap(model, X_row)
    raw_sv = explainer.shap_values(X_row)
    shap_vals = _extract_shap(raw_sv)[0]

    factors = []
    for i, name in enumerate(feature_names):
        impact = shap_vals[i]
        factors.append(
            {
                "feature": name,
                "value": X_row.iloc[0, i],
                "impact": float(impact),
                "importance": "high" if abs(impact) > np.percentile(np.abs(shap_vals), 75) else "medium" if abs(impact) > np.percentile(np.abs(shap_vals), 50) else "low",
            }
        )

    factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return factors[:top_n]


def generate_retention_suggestion(
    factors: list[dict], churn_prob: float, raw_features: dict = None
) -> str:
    suggestions = []

    for f in factors:
        feat = f["feature"]

        if raw_features is not None:
            raw_val = raw_features.get(feat, None)
        else:
            raw_val = f["value"]

        if feat == "Contract" and raw_val in ("Month-to-month", 0):
            suggestions.append("Offer an annual or two-year contract with a discount")

        elif feat == "tenure":
            if raw_features is not None and isinstance(raw_val, (int, float)) and 0 <= raw_val <= 72:
                if raw_val < 12:
                    suggestions.append(
                        f"Customer has low tenure ({int(raw_val)} months) - consider a loyalty reward or onboarding program"
                    )
            elif raw_features is None and isinstance(raw_val, (int, float)) and raw_val < -0.5:
                suggestions.append("Customer has low tenure - consider a loyalty reward or onboarding program")

        elif feat == "MonthlyCharges":
            if raw_features is not None and isinstance(raw_val, (int, float)) and 0 < raw_val < 500:
                if raw_val > 70:
                    suggestions.append(
                        f"Monthly charges are high (${raw_val:.0f}) - consider a tailored discount or plan downgrade"
                    )
            elif raw_features is None and isinstance(raw_val, (int, float)) and raw_val > 0.5:
                suggestions.append("Monthly charges are high - consider a tailored discount or plan downgrade")

        elif feat == "InternetService" and raw_val in ("Fiber optic", 1):
            suggestions.append("Fiber optic users have higher churn — check service quality and offer a credit")

        elif feat == "PaymentMethod" and raw_val in ("Electronic check", 0):
            suggestions.append("Electronic check users churn more — offer autopay discount")

        elif feat == "PaperlessBilling" and raw_val in ("Yes", 1):
            suggestions.append("Paperless billing users churn more — offer a small incentive to stay")

        elif feat == "OnlineSecurity" and raw_val in ("No", 0):
            suggestions.append("Customer lacks online security — bundle it free for 3 months")

        elif feat == "TechSupport" and raw_val in ("No", 0):
            suggestions.append("Customer has no tech support — offer a free trial of tech support")

    if not suggestions:
        suggestions.append("Continue monitoring — churn risk is driven by general factors")

    top = suggestions[:3]
    return " | ".join(top)


def interpret_prediction(
    model, X_row: pd.DataFrame, feature_names: list[str], raw_features: dict = None
) -> dict:
    proba = model.predict_proba(X_row)[0, 1]
    prediction = "Yes" if proba >= 0.5 else "No"
    top_factors = get_top_factors(model, X_row, feature_names)
    suggestion = generate_retention_suggestion(top_factors, proba, raw_features)

    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction": prediction,
        "top_factors": top_factors,
        "retention_suggestion": suggestion,
    }
