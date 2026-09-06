"""Trains and compares multiple salary-prediction models, picks the
best on validation R^2, and saves it + preprocessing artifacts with joblib.
Run: python scripts/train_salary_model.py
"""
import ast
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "jobs_processed.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL = ["title_clean", "location_clean", "experience_level", "industry", "education", "employment_type"]
NUMERIC = ["experience_years", "n_skills"]


def build_features(df: pd.DataFrame):
    df = df.copy()
    df["skills_extracted"] = df["skills_extracted"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else s
    )
    mlb = MultiLabelBinarizer()
    skill_matrix = mlb.fit_transform(df["skills_extracted"])
    skill_df = pd.DataFrame(skill_matrix, columns=[f"skill_{c}" for c in mlb.classes_], index=df.index)
    X = pd.concat([df[CATEGORICAL + NUMERIC], skill_df], axis=1)
    y = df["salary"].values
    return X, y, mlb


def main():
    df = pd.read_csv(DATA_PATH)
    if "experience_years" not in df.columns:
        df["experience_years"] = 2
    X, y, mlb = build_features(df)

    skill_cols = [c for c in X.columns if c.startswith("skill_")]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="passthrough",  # numeric + skill columns pass through
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0),
    }

    results = {}
    best_name, best_pipe, best_r2 = None, None, -np.inf

    for name, model in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = r2_score(y_test, preds)
        mape = float(np.mean(np.abs((y_test - preds) / y_test)) * 100)
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
        print(f"{name}: MAE={mae:.0f} RMSE={rmse:.0f} R2={r2:.3f} MAPE={mape:.1f}%")
        if r2 > best_r2:
            best_r2, best_name, best_pipe = r2, name, pipe

    print(f"\nBest model: {best_name} (R2={best_r2:.3f})")

    # Residual std for a simple uncertainty band on predictions
    residuals = y_test - best_pipe.predict(X_test)
    residual_std = float(np.std(residuals))

    joblib.dump(best_pipe, MODEL_DIR / "salary_model.joblib")
    joblib.dump(mlb, MODEL_DIR / "skill_binarizer.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")

    with open(MODEL_DIR / "salary_model_metrics.json", "w") as f:
        json.dump({"best_model": best_name, "residual_std": residual_std, "all_results": results}, f, indent=2)

    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
