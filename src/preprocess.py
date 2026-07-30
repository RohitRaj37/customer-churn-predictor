import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import joblib
import os


PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_raw(path: str = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def encode_features(
    df: pd.DataFrame, fit: bool = True, encoders: dict = None
) -> tuple[pd.DataFrame, dict]:
    ordinal_map = {"Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}}
    for col, mapping in ordinal_map.items():
        df[col] = df[col].map(mapping)

    nominal_cols = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    if fit:
        encoders = {}
        for col in nominal_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in nominal_cols:
            df[col] = encoders[col].transform(df[col])

    return df, encoders


def scale_features(
    df: pd.DataFrame,
    fit: bool = True,
    scaler: StandardScaler = None,
    scale_cols: list[str] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    if scale_cols is None:
        scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    if fit:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        df[scale_cols] = scaler.transform(df[scale_cols])
    return df, scaler


def split_data(
    df: pd.DataFrame, target: str = "Churn", test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(target, axis=1)
    y = df[target].map({"No": 0, "Yes": 1})
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def preprocess(
    df: pd.DataFrame = None,
    fit: bool = True,
    encoders: dict = None,
    scaler: StandardScaler = None,
    save: bool = True,
) -> dict:
    if df is None:
        df = load_raw()
    df = clean(df)
    df, encoders = encode_features(df, fit=fit, encoders=encoders)
    df, scaler = scale_features(df, fit=fit, scaler=scaler)

    if save and fit:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        df.to_csv(PROCESSED_DIR / "processed.csv", index=False)
        joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    X_train, X_test, y_train, y_test = split_data(df)

    if save:
        for name, obj in [
            ("X_train", X_train),
            ("X_test", X_test),
            ("y_train", y_train),
            ("y_test", y_test),
        ]:
            joblib.dump(obj, PROCESSED_DIR / f"{name}.pkl")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "scaler": scaler,
        "feature_names": X_train.columns.tolist(),
    }


def load_preprocessed() -> dict:
    X_train = joblib.load(PROCESSED_DIR / "X_train.pkl")
    return {
        "X_train": X_train,
        "X_test": joblib.load(PROCESSED_DIR / "X_test.pkl"),
        "y_train": joblib.load(PROCESSED_DIR / "y_train.pkl"),
        "y_test": joblib.load(PROCESSED_DIR / "y_test.pkl"),
        "encoders": joblib.load(MODELS_DIR / "encoders.pkl"),
        "scaler": joblib.load(MODELS_DIR / "scaler.pkl"),
        "feature_names": X_train.columns.tolist(),
    }


if __name__ == "__main__":
    result = preprocess()
    print("Preprocessing complete.")
    print(f"X_train: {result['X_train'].shape}")
    print(f"X_test: {result['X_test'].shape}")
