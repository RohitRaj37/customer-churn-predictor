import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from src.preprocess import encode_features, scale_features
from src.interpret import interpret_prediction, plot_shap_waterfall

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@st.cache_resource
def load_artifacts():
    scores = joblib.load(MODELS_DIR / "scores.pkl")
    best_name = max(scores, key=scores.get)
    model = joblib.load(MODELS_DIR / f"{best_name}.pkl")
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = joblib.load(PROCESSED_DIR / "X_train.pkl").columns.tolist()
    return model, best_name, encoders, scaler, feature_names


model, model_name, encoders, scaler, feature_names = load_artifacts()


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Customer Churn Predictor")
st.markdown(
    f"**Best Model:** `{model_name}` | **Features:** {len(feature_names)}"
)

tab1, tab2 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction"])


with tab1:
    st.header("Single Customer Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
        total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

    if st.button("Predict Churn", type="primary", use_container_width=True):
        raw = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        row = pd.DataFrame([raw])
        row, _ = encode_features(row, fit=False, encoders=encoders)
        row, _ = scale_features(row, fit=False, scaler=scaler)
        row = row[feature_names]

        result = interpret_prediction(model, row, feature_names, raw_features=raw)

        prob = result["churn_probability"]
        pred = result["churn_prediction"]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            color = "red" if pred == "Yes" else "green"
            st.metric("Prediction", pred, delta=None)
            st.markdown(
                f"<h1 style='color: {color}; text-align: center;'>{pred}</h1>",
                unsafe_allow_html=True,
            )
        with col_b:
            st.metric("Churn Probability", f"{prob:.1%}")
            st.progress(prob)
        with col_c:
            st.metric("Retention Suggestion", result["retention_suggestion"][:50] + "...")

        st.subheader("Top Factors Driving Prediction")
        for f in result["top_factors"]:
            icon = "🔴" if f["impact"] > 0 else "🟢"
            st.markdown(
                f"{icon} **{f['feature']}** (value={f['value']}, impact={f['impact']:.3f})"
            )

        with st.expander("SHAP Waterfall Plot"):
            fig = plot_shap_waterfall(model, row, idx=0)
            st.pyplot(fig)
            plt.close("all")


with tab2:
    st.header("Batch Prediction from CSV")

    uploaded_file = st.file_uploader("Upload CSV with customer data", type=["csv"])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Uploaded {len(df_batch)} rows. Preview:")
        st.dataframe(df_batch.head())

        if st.button("Run Batch Prediction", type="primary"):
            df_input = df_batch.copy()
            if "customerID" in df_input.columns:
                ids = df_input["customerID"]
                df_input = df_input.drop("customerID", axis=1)
            else:
                ids = pd.RangeIndex(len(df_input))

            if "TotalCharges" in df_input.columns:
                df_input["TotalCharges"] = pd.to_numeric(df_input["TotalCharges"], errors="coerce")
                df_input["TotalCharges"].fillna(
                    joblib.load(PROCESSED_DIR / "X_train.pkl")["TotalCharges"].median(),
                    inplace=True,
                )
            if "SeniorCitizen" in df_input.columns:
                df_input["SeniorCitizen"] = df_input["SeniorCitizen"].map({0: "No", 1: "Yes"})

            df_input, _ = encode_features(df_input, fit=False, encoders=encoders)
            df_input, _ = scale_features(df_input, fit=False, scaler=scaler)
            df_input = df_input[feature_names]

            y_proba = model.predict_proba(df_input)[:, 1]
            y_pred = model.predict(df_input)

            results = []
            for i in range(len(df_input)):
                raw_row = df_input.iloc[[i]]
                interpretation = interpret_prediction(model, raw_row, feature_names)
                top_reason = interpretation["top_factors"][0]["feature"] if interpretation["top_factors"] else ""
                results.append(
                    {
                        "customerID": ids.iloc[i] if hasattr(ids, "iloc") else ids[i],
                        "churn_prediction": y_pred[i] if isinstance(y_pred[i], str) else ("Yes" if y_pred[i] == 1 else "No"),
                        "churn_probability": round(float(y_proba[i]), 4),
                        "top_churn_reason": top_reason,
                        "retention_suggestion": interpretation["retention_suggestion"],
                    }
                )

            df_results = pd.DataFrame(results)
            st.write("Prediction Results:")
            st.dataframe(df_results)

            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results (CSV)",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

            churn_count = (df_results["churn_prediction"] == "Yes").sum()
            st.info(
                f"Predicted churners: **{churn_count} / {len(df_results)}** "
                f"({churn_count / len(df_results) * 100:.1f}%)"
            )
