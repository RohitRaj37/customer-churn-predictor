# Customer Churn Predictor

Predict which customers are about to leave, explain why with SHAP, and suggest retention actions.

## Project Overview

| Component | Description |
|-----------|-------------|
| **Dataset** | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telecom-churn) — 7,043 customers, 21 features |
| **ML Models** | Logistic Regression, Random Forest, XGBoost, Voting Ensemble |
| **Best ROC-AUC** | ~0.86 |
| **Interpretation** | SHAP summary + waterfall plots for individual predictions |
| **API** | FastAPI endpoint (`/predict`) with explanations |
| **Dashboard** | Streamlit app for single & batch predictions |

## Project Structure

```
customer-churn-predictor/
├── data/                   # raw + processed data
│   └── fetch_data.py       # download script
├── notebooks/
│   └── 01_eda.ipynb        # exploratory analysis
├── src/
│   ├── eda.py              # EDA helper functions
│   ├── preprocess.py       # cleaning, encoding, scaling
│   ├── train.py            # model training + tuning
│   ├── evaluate.py         # metrics + plots
│   └── interpret.py        # SHAP explanations + retention logic
├── models/                 # saved .pkl files
├── api/
│   └── app.py              # FastAPI service
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── run_pipeline.py         # end-to-end pipeline script
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py --all
```

This will:
- Download the Telco dataset
- Clean, encode, and scale features
- Train 4 models and save the best one
- Evaluate and save ROC/PR curves
- Run SHAP interpretation on sample predictions

### 3. Launch the API

```bash
python -m api.app
```

Then send a prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "Contract": "Month-to-month", "MonthlyCharges": 85.0}'
```

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| **EDA** | `src/eda.py` | Target distribution, feature analysis, correlation heatmaps |
| **Preprocess** | `src/preprocess.py` | Handle missing values, encode categories, scale numerics |
| **Train** | `src/train.py` | Train 4 models, optional GridSearchCV tuning |
| **Evaluate** | `src/evaluate.py` | Metrics, confusion matrix, ROC & PR curves |
| **Interpret** | `src/interpret.py` | SHAP explanations, top factors, retention suggestions |

## Key Findings

- **Contract type** is the strongest predictor — month-to-month contracts churn at ~43%
- **Tenure** — customers with < 12 months have high churn risk
- **Monthly charges** — higher charges correlate with higher churn
- **Fiber optic + Electronic check** — combination signals elevated risk

## Resume Bullet Points

> **Customer Churn Predictor** — Built an end-to-end ML pipeline (EDA → feature engineering → XGBoost → SHAP interpretation). Deployed via FastAPI + Streamlit dashboard with "what-if" retention suggestions. Achieved 0.86 ROC-AUC on Telco dataset.

## License

MIT
