import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "all_month.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_plot(filename: str):
    """Save current matplotlib figure into static/plots."""
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=200)
    plt.close()


def plot_before_training(raw_df: pd.DataFrame):
    """Visualizations BEFORE cleaning/training (raw dataset)."""

    # Missing values (top 15)
    missing = raw_df.isna().sum().sort_values(ascending=False)
    plt.figure()
    missing.head(15).plot(kind="bar")
    plt.title("Missing Values (Top 15 Columns) - BEFORE Cleaning")
    plt.xlabel("Columns")
    plt.ylabel("Missing count")
    save_plot("before_missing_values.png")

    # Magnitude histogram
    if "mag" in raw_df.columns:
        plt.figure()
        raw_df["mag"].dropna().plot(kind="hist", bins=40)
        plt.title("Magnitude Distribution - BEFORE Cleaning")
        plt.xlabel("Magnitude (mag)")
        plt.ylabel("Frequency")
        save_plot("before_mag_hist.png")

    # Depth vs Magnitude scatter
    if {"depth", "mag"}.issubset(raw_df.columns):
        plt.figure()
        plt.scatter(raw_df["depth"], raw_df["mag"], s=8)
        plt.title("Depth vs Magnitude - BEFORE Cleaning")
        plt.xlabel("Depth (km)")
        plt.ylabel("Magnitude (mag)")
        save_plot("before_depth_vs_mag.png")


def plot_after_cleaning(clean_df: pd.DataFrame):
    """Visualizations AFTER cleaning (correlation heatmap)."""
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = clean_df[numeric_cols].corr(numeric_only=True)

        plt.figure(figsize=(10, 8))
        plt.imshow(corr.values, aspect="auto")
        plt.title("Correlation Heatmap (Numeric Features) - AFTER Cleaning")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.colorbar()
        save_plot("after_corr_heatmap.png")


def plot_after_training(y_true, y_pred, title_prefix: str, filename_prefix: str):
    """Plots AFTER training (actual vs predicted + residuals)."""
    # Actual vs Predicted
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    plt.title(f"{title_prefix}: Actual vs Predicted")
    plt.xlabel("Actual magnitude")
    plt.ylabel("Predicted magnitude")
    save_plot(f"{filename_prefix}_actual_vs_pred.png")

    # Residual plot
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, s=10)
    plt.axhline(0)
    plt.title(f"{title_prefix}: Residual Plot")
    plt.xlabel("Predicted magnitude")
    plt.ylabel("Residual (Actual - Predicted)")
    save_plot(f"{filename_prefix}_residuals.png")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH)

    # BEFORE plots
    plot_before_training(raw_df)

    df = raw_df.drop_duplicates().copy()

    # convert numeric columns safely
    numeric_candidates = [
        "latitude", "longitude", "depth", "mag",
        "nst", "gap", "dmin", "rms",
        "horizontalError", "depthError", "magError", "magNst"
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    target = "mag"

    feature_cols = [
        "latitude", "longitude", "depth",
        "nst", "gap", "dmin", "rms",
        "horizontalError", "depthError", "magError", "magNst"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # drop rows where target missing
    df = df.dropna(subset=[target])

    # keep only needed columns
    model_df = df[feature_cols + [target]].copy()

    # AFTER cleaning plots
    plot_after_cleaning(model_df)

    # Train/Test
    X = model_df[feature_cols]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=7),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
    }

    pipelines = {}
    metrics = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        pipelines[name] = pipe
        metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        # AFTER training plots per model
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_after_training(
            y_true=y_test.values,
            y_pred=y_pred,
            title_prefix=name,
            filename_prefix=f"after_{safe_name}"
        )

        # Feature importance for Random Forest
        if name == "Random Forest":
            try:
                rf = pipe.named_steps["model"]
                importances = rf.feature_importances_
                plt.figure()
                plt.bar(feature_cols, importances)
                plt.title("Random Forest Feature Importance")
                plt.xlabel("Features")
                plt.ylabel("Importance")
                plt.xticks(rotation=90)
                save_plot("after_rf_feature_importance.png")
            except Exception:
                pass

    # Choose best model by lowest RMSE
    best_model_name = min(metrics.keys(), key=lambda k: metrics[k]["RMSE"])
    best_rmse = metrics[best_model_name]["RMSE"]

    # Save model pack (keys match app.py)
    model_pack = {
        "feature_cols": feature_cols,
        "best_model_name": best_model_name,
        "pipelines": pipelines,
    }
    joblib.dump(model_pack, os.path.join(MODEL_DIR, "models.pkl"))

    # Save metrics pack (contains "metrics")
    metrics_pack = {
        "metrics": metrics,
        "best_model_name": best_model_name,
        "best_model_rmse": best_rmse,
        "feature_cols": feature_cols,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_pack, f, indent=2)

    # Model comparison RMSE bar chart
    model_names = list(metrics.keys())
    rmses = [metrics[m]["RMSE"] for m in model_names]
    plt.figure()
    plt.bar(model_names, rmses)
    plt.title("Model Comparison (RMSE) - LOWER is better")
    plt.xlabel("Models")
    plt.ylabel("RMSE")
    plt.xticks(rotation=15)
    save_plot("after_model_comparison_rmse.png")

    print("\n Training complete!")
    print(f"Best model: {best_model_name} (RMSE: {best_rmse:.4f})")
    print("Saved: model/models.pkl, model/metrics.json")
    print("Plots: static/plots/")


if __name__ == "__main__":
    main()
