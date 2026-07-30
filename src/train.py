import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_logistic_regression(X_train, y_train, **kwargs):
    params = {
        "C": kwargs.get("C", 1.0),
        "max_iter": kwargs.get("max_iter", 1000),
        "solver": kwargs.get("solver", "liblinear"),
        "random_state": 42,
        "class_weight": "balanced",
    }
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model, params


def train_random_forest(X_train, y_train, **kwargs):
    params = {
        "n_estimators": kwargs.get("n_estimators", 300),
        "max_depth": kwargs.get("max_depth", 10),
        "min_samples_split": kwargs.get("min_samples_split", 10),
        "min_samples_leaf": kwargs.get("min_samples_leaf", 4),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model, params


def train_xgboost(X_train, y_train, **kwargs):
    y_train_bin = (y_train == "Yes").astype(int) if y_train.dtype == "object" else y_train
    scale_pos_weight = (y_train_bin == 0).sum() / (y_train_bin == 1).sum()
    params = {
        "n_estimators": kwargs.get("n_estimators", 200),
        "max_depth": kwargs.get("max_depth", 5),
        "learning_rate": kwargs.get("learning_rate", 0.05),
        "subsample": kwargs.get("subsample", 0.8),
        "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
        "scale_pos_weight": kwargs.get("scale_pos_weight", scale_pos_weight),
        "eval_metric": "auc",
        "random_state": 42,
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model, params


def train_voting_ensemble(X_train, y_train, models: list = None) -> VotingClassifier:
    if models is None:
        rf, _ = train_random_forest(X_train, y_train, n_estimators=200, max_depth=8)
        xgb_model, _ = train_xgboost(X_train, y_train, n_estimators=150, max_depth=4)
    else:
        rf, xgb_model = models

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_model)],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    return ensemble


def tune_xgboost(X_train, y_train, X_test, y_test) -> xgb.XGBClassifier:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }
    y_train_bin = (y_train == "Yes").astype(int) if y_train.dtype == "object" else y_train
    scale_pos_weight = (y_train_bin == 0).sum() / (y_train_bin == 1).sum()
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    grid = GridSearchCV(
        model,
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"Best XGBoost params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_all(X_train, y_train, X_test, y_test, tune: bool = False) -> dict:
    results = {}

    print("Training Logistic Regression...")
    lr, lr_params = train_logistic_regression(X_train, y_train)
    lr_score = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    results["LogisticRegression"] = {"model": lr, "params": lr_params, "roc_auc": lr_score}
    print(f"  ROC-AUC: {lr_score:.4f}")

    print("Training Random Forest...")
    rf, rf_params = train_random_forest(X_train, y_train)
    rf_score = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    results["RandomForest"] = {"model": rf, "params": rf_params, "roc_auc": rf_score}
    print(f"  ROC-AUC: {rf_score:.4f}")

    if tune:
        print("Tuning XGBoost with GridSearchCV...")
        xgb_model = tune_xgboost(X_train, y_train, X_test, y_test)
    else:
        print("Training XGBoost...")
        xgb_model, xgb_params = train_xgboost(X_train, y_train)

    xgb_score = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    results["XGBoost"] = {"model": xgb_model, "roc_auc": xgb_score}
    print(f"  ROC-AUC: {xgb_score:.4f}")

    print("Training Voting Ensemble (RF + XGB)...")
    ensemble = train_voting_ensemble(
        X_train, y_train, models=[results["RandomForest"]["model"], results["XGBoost"]["model"]]
    )
    ensemble_score = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1])
    results["VotingEnsemble"] = {"model": ensemble, "roc_auc": ensemble_score}
    print(f"  ROC-AUC: {ensemble_score:.4f}")

    return results


def save_models(results: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, info in results.items():
        path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(info["model"], path)
        print(f"Saved {name} -> {path}")
    scores = {name: info["roc_auc"] for name, info in results.items()}
    joblib.dump(scores, MODELS_DIR / "scores.pkl")
    print(f"Saved scores -> {MODELS_DIR / 'scores.pkl'}")


def load_best_model() -> tuple:
    scores = joblib.load(MODELS_DIR / "scores.pkl")
    best_name = max(scores, key=scores.get)
    model = joblib.load(MODELS_DIR / f"{best_name}.pkl")
    print(f"Loaded best model: {best_name} (ROC-AUC: {scores[best_name]:.4f})")
    return model, best_name


if __name__ == "__main__":
    from src.preprocess import load_preprocessed

    data = load_preprocessed()
    results = train_all(data["X_train"], data["y_train"], data["X_test"], data["y_test"])
    save_models(results)
