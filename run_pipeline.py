import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Predictor — Full Pipeline")
    parser.add_argument("--fetch", action="store_true", help="Download raw data")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--tune", action="store_true", help="Tune XGBoost with GridSearchCV")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models and plot metrics")
    parser.add_argument("--interpret", action="store_true", help="Run SHAP interpretation")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline")
    args = parser.parse_args()

    if args.all or args.fetch:
        print("=" * 60)
        print("STEP 1: Fetching data")
        print("=" * 60)
        from data.fetch_data import download
        download()
        print()

    if args.all or args.preprocess:
        print("=" * 60)
        print("STEP 2: Preprocessing")
        print("=" * 60)
        from src.preprocess import preprocess
        preprocess()
        print()

    if args.all or args.train or args.tune:
        print("=" * 60)
        print("STEP 3: Training models")
        print("=" * 60)
        from src.preprocess import load_preprocessed
        from src.train import train_all, save_models
        data = load_preprocessed()
        results = train_all(
            data["X_train"], data["y_train"],
            data["X_test"], data["y_test"],
            tune=(args.tune or args.all),
        )
        save_models(results)
        print()

    if args.all or args.evaluate:
        print("=" * 60)
        print("STEP 4: Evaluation")
        print("=" * 60)
        from src.preprocess import load_preprocessed
        from src.train import load_best_model
        import joblib
        data = load_preprocessed()
        scores = joblib.load(Path("models") / "scores.pkl")
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  {name:20s}  ROC-AUC: {score:.4f}")
        best_model, best_name = load_best_model()
        from src.evaluate import plot_roc_curve, plot_pr_curve, plot_confusion_matrix
        fig1 = plot_roc_curve(best_model, data["X_test"], data["y_test"])
        fig1.savefig("models/roc_curve.png", dpi=150, bbox_inches="tight")
        fig2 = plot_pr_curve(best_model, data["X_test"], data["y_test"])
        fig2.savefig("models/pr_curve.png", dpi=150, bbox_inches="tight")
        fig3 = plot_confusion_matrix(best_model, data["X_test"], data["y_test"])
        fig3.savefig("models/confusion_matrix.png", dpi=150, bbox_inches="tight")
        print("Plots saved to models/")
        print()

    if args.all or args.interpret:
        print("=" * 60)
        print("STEP 5: SHAP Interpretation")
        print("=" * 60)
        from src.preprocess import load_preprocessed
        from src.train import load_best_model
        from src.interpret import plot_shap_summary, interpret_prediction
        import joblib
        from pathlib import Path
        data = load_preprocessed()
        best_model, best_name = load_best_model()
        shap_model = best_model
        shap_name = best_name
        if "Ensemble" in best_name or "Voting" in best_name:
            for fallback in ["RandomForest", "XGBoost"]:
                fb_path = Path("models") / f"{fallback}.pkl"
                if fb_path.exists():
                    shap_model = joblib.load(fb_path)
                    shap_name = fallback
                    print(f"Using {fallback} for SHAP explanations (instead of {best_name})")
                    break
        print(f"Generating SHAP summary plot...")
        fig = plot_shap_summary(shap_model, data["X_test"])
        fig.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
        print("Saved models/shap_summary.png")
        sample = data["X_test"].iloc[:5]
        for idx in range(len(sample)):
            row = sample.iloc[[idx]]
            result = interpret_prediction(shap_model, row, data["feature_names"])
            print(f"\n  Customer #{idx}:")
            print(f"    Churn Probability: {result['churn_probability']:.2%}")
            print(f"    Prediction: {result['churn_prediction']}")
            print(f"    Top factors: {[f['feature'] for f in result['top_factors']]}")
            print(f"    Suggestion: {result['retention_suggestion']}")
        print()

    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
