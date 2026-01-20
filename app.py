import os
import json
import joblib
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "models.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

app = Flask(__name__)


def mag_category(m):
    if m < 3.0: return "Minor (usually not felt)"
    if m < 4.0: return "Light (often felt, little damage)"
    if m < 5.0: return "Moderate (some damage possible)"
    if m < 6.0: return "Strong (damage likely)"
    if m < 7.0: return "Major (serious damage)"
    return "Great (severe damage potential)"


def validate_float(value, field_name, min_v=None, max_v=None):
    if value is None or str(value).strip() == "":
        return None, f"{field_name} is required."
    try:
        v = float(value)
    except ValueError:
        return None, f"{field_name} must be a number."

    if min_v is not None and v < min_v:
        return None, f"{field_name} must be ≥ {min_v}."
    if max_v is not None and v > max_v:
        return None, f"{field_name} must be ≤ {max_v}."
    return v, None


# Load model pack
if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
    raise FileNotFoundError(
        "models.pkl or metrics.json not found. Run: python train_and_analyze.py first."
    )

model_pack = joblib.load(MODEL_PATH)
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics_pack = json.load(f)

FEATURE_COLS = model_pack["feature_cols"]
PIPELINES = model_pack["pipelines"]
BEST_MODEL_NAME = model_pack["best_model_name"]

METRICS = metrics_pack["metrics"]  # dict: model -> {MAE, RMSE, R2}


@app.route("/")
def index():
    return render_template("index.html", best_model=BEST_MODEL_NAME)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model_names = list(PIPELINES.keys())

    if request.method == "GET":
        return render_template(
            "predict.html",
            feature_cols=FEATURE_COLS,
            model_names=model_names,
            best_model=BEST_MODEL_NAME,
            metrics=METRICS
        )

    # POST
    selected_model = request.form.get("model_name", BEST_MODEL_NAME)
    if selected_model not in PIPELINES:
        selected_model = BEST_MODEL_NAME

    # Range rules (only apply if column exists)
    rules = {
        "latitude": (-90, 90),
        "longitude": (-180, 180),
        "depth": (0, 700),
        "gap": (0, 360),
        "rms": (0, 5),
        "dmin": (0, 50),
        "nst": (0, 500),
        "magError": (0, 5),
        "depthError": (0, 200),
        "horizontalError": (0, 200),
        "magNst": (0, 1000),
    }

    inputs, errors = [], []
    for col in FEATURE_COLS:
        lo, hi = rules.get(col, (None, None))
        val, err = validate_float(request.form.get(col), col, lo, hi)
        if err:
            errors.append(err)
        inputs.append(val)

    if errors:
        return render_template(
            "predict.html",
            feature_cols=FEATURE_COLS,
            model_names=model_names,
            best_model=BEST_MODEL_NAME,
            metrics=METRICS,
            errors=errors,
            old=request.form
        )

    X = np.array(inputs, dtype=float).reshape(1, -1)
    pipeline = PIPELINES[selected_model]

    pred = float(pipeline.predict(X)[0])
    pred_round = round(pred, 2)

    category = mag_category(pred)

    rmse = float(METRICS[selected_model]["RMSE"])
    low = round(pred - rmse, 2)
    high = round(pred + rmse, 2)

    return render_template(
        "result.html",
        prediction=pred_round,
        category=category,
        model_name=selected_model,
        rmse=rmse,
        low=low,
        high=high,
        metrics=METRICS[selected_model]
    )


@app.route("/analysis")
def analysis():
    plots_dir = os.path.join(BASE_DIR, "static", "plots")
    plot_files = []
    if os.path.exists(plots_dir):
        plot_files = sorted([f for f in os.listdir(plots_dir) if f.lower().endswith(".png")])

    return render_template(
        "analysis.html",
        plot_files=plot_files,
        best_model=BEST_MODEL_NAME,
        metrics=METRICS
    )


if __name__ == "__main__":
    app.run(debug=True)
