import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(path: str = None) -> pd.DataFrame:
    if path is None:
        path = str(
            Path(__file__).resolve().parent.parent
            / "data"
            / "raw"
            / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.drop("customerID", axis=1, inplace=True)
    return df


def basic_stats(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "churn_rate": df["Churn"].value_counts(normalize=True).to_dict(),
    }


def plot_target_distribution(df: pd.DataFrame, ax=None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Churn"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{val} ({val / len(df) * 100:.1f}%)",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title("Target Distribution (Churn)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Churn")
    sns.despine()
    return ax.figure


def plot_numeric_distributions(
    df: pd.DataFrame, cols: list[str] = None, ax=None
) -> plt.Figure:
    if cols is None:
        cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    for i, col in enumerate(cols):
        sns.histplot(
            data=df,
            x=col,
            hue="Churn",
            kde=True,
            palette=["#2ecc71", "#e74c3c"],
            alpha=0.6,
            ax=axes[0, i],
        )
        axes[0, i].set_title(f"{col} Distribution by Churn")
        sns.boxplot(
            data=df,
            x="Churn",
            y=col,
            palette=["#2ecc71", "#e74c3c"],
            ax=axes[1, i],
        )
        axes[1, i].set_title(f"{col} Boxplot by Churn")
    plt.tight_layout()
    return fig


def plot_categorical_churn_rates(
    df: pd.DataFrame, cols: list[str] = None, ax=None
) -> plt.Figure:
    if cols is None:
        cols = [
            "Contract",
            "InternetService",
            "PaymentMethod",
            "PaperlessBilling",
            "Partner",
            "Dependents",
        ]
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        rates = df.groupby(col)["Churn"].value_counts(normalize=True).unstack() * 100
        rates.plot(kind="bar", stacked=True, ax=axes[i], color=["#2ecc71", "#e74c3c"])
        axes[i].set_title(f"Churn Rate by {col}", fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Percentage")
        axes[i].legend(title="Churn", loc="upper right")
        axes[i].tick_params(axis="x", rotation=45)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, ax=None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    return ax.figure
