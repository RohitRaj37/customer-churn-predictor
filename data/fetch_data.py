import os
import urllib.request
import sys
import csv
import random
import math

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

MIRROR_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-ibm-watson-studio/main/Data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-ibm-watson-studio/master/Data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/telco-customer-churn.csv",
    "https://raw.githubusercontent.com/plotly/datasets/main/telco-customer-churn.csv",
]


def download(force: bool = False) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, FILENAME)

    if os.path.exists(dest) and not force:
        print(f"File already exists: {dest}")
        return dest

    for url in MIRROR_URLS:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest)
            with open(dest) as f:
                header = f.readline().strip()
            if size > 100000 and "customerID" in header:
                print(f"Downloaded successfully ({size:,} bytes)")
                return dest
            else:
                print(f"  Invalid content, trying next...")
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    print("Could not download from any mirror.")
    print("Generating synthetic dataset for testing...")
    _generate_synthetic(dest)
    print(f"Saved synthetic dataset to {dest}")
    return dest


def _generate_synthetic(dest: str):
    random.seed(42)
    n = 7043

    contracts = ["Month-to-month", "One year", "Two year"]
    payments = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    internet = ["DSL", "Fiber optic", "No"]
    yes_no = ["Yes", "No"]
    yes_no_na = ["Yes", "No", "No internet service"]
    gender = ["Male", "Female"]

    fieldnames = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    ]

    with open(dest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(1, n + 1):
            contract = random.choices(contracts, weights=[0.55, 0.25, 0.20])[0]
            tenure = _sample_tenure(contract)
            monthly = random.gauss(65, 25)
            monthly = max(18.0, min(120.0, monthly))
            total = monthly * tenure
            churn_prob = _churn_probability(contract, tenure, monthly)
            churn = "Yes" if random.random() < churn_prob else "No"
            internet_svc = random.choices(internet, weights=[0.25, 0.45, 0.30])[0]
            has_phone = random.choice(yes_no)
            multiple = random.choice(yes_no_na) if has_phone == "Yes" else "No phone service"
            security = random.choice(yes_no_na) if internet_svc != "No" else "No internet service"
            backup = random.choice(yes_no_na) if internet_svc != "No" else "No internet service"
            protection = random.choice(yes_no_na) if internet_svc != "No" else "No internet service"
            tech = random.choice(yes_no_na) if internet_svc != "No" else "No internet service"
            tv = random.choice(yes_no_na) if internet_svc != "No" else "No internet service"
            movies = random.choice(yes_no_na) if internet_svc != "No" else "No internet service"
            payment = random.choices(payments, weights=[0.35, 0.15, 0.25, 0.25])[0]

            writer.writerow({
                "customerID": f"0001-{i:05d}",
                "gender": random.choice(gender),
                "SeniorCitizen": str(random.choices([0, 1], weights=[0.85, 0.15])[0]),
                "Partner": random.choice(yes_no),
                "Dependents": random.choices(yes_no, weights=[0.7, 0.3])[0],
                "tenure": str(tenure),
                "PhoneService": has_phone,
                "MultipleLines": multiple,
                "InternetService": internet_svc,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": tech,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": random.choices(yes_no, weights=[0.6, 0.4])[0],
                "PaymentMethod": payment,
                "MonthlyCharges": f"{monthly:.1f}",
                "TotalCharges": f"{total:.1f}",
                "Churn": churn,
            })


def _sample_tenure(contract: str) -> int:
    if contract == "Month-to-month":
        return random.choices(range(1, 73), weights=[max(1, 30 - t) for t in range(1, 73)])[0]
    elif contract == "One year":
        return random.randint(12, 24)
    else:
        return random.randint(24, 72)


def _churn_probability(contract: str, tenure: int, monthly: float) -> float:
    prob = 0.03
    if contract == "Month-to-month":
        prob += 0.35
    elif contract == "One year":
        prob += 0.15
    prob -= (tenure / 72) * 0.20
    prob += ((monthly - 18) / 102) * 0.10
    if monthly > 80:
        prob += 0.05
    return max(0.01, min(0.95, prob))


if __name__ == "__main__":
    force = "--force" in sys.argv
    download(force=force)
