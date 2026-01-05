from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

crop_model = joblib.load(BASE_DIR / "crop_pred_new_model.pkl")
fert_model = joblib.load(BASE_DIR / "fertilizer_model.pkl")

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FERT_FEATURES = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Soil Type",
    "Crop Type",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
]


def _extract_payload(expected_fields) -> Tuple[Dict, Dict]:
    """
    Pull JSON or form data into a dict, coercing strings to numbers where possible
    and returning both the cleaned payload and any validation errors.
    """
    raw_data = request.get_json(silent=True) or request.form.to_dict()
    payload, errors = {}, {}

    for field, caster in expected_fields.items():
        incoming_key = field if field in raw_data else field.lower()
        value = raw_data.get(incoming_key)

        if value in (None, ""):
            errors[field] = "Missing value"
            continue

        try:
            payload[field] = caster(value)
        except Exception:
            errors[field] = f"Invalid value for {field}"

    return payload, errors


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/guides", methods=["GET"])
def guides():
    return render_template("guides.html")


@app.route("/weather", methods=["GET"])
def weather():
    return render_template("weather.html")


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.route("/api/crop", methods=["POST"])
def predict_crop():
    expected = {
        "N": int,
        "P": int,
        "K": int,
        "temperature": float,
        "humidity": float,
        "ph": float,
        "rainfall": float,
    }

    payload, errors = _extract_payload(expected)
    if errors:
        return jsonify({"error": errors}), 400

    if payload["temperature"] < -10 or payload["temperature"] > 60:
        return jsonify({"error": {"temperature": "Range must be between -10 and 60 C"}}), 400
    if payload["humidity"] < 0 or payload["humidity"] > 100:
        return jsonify({"error": {"humidity": "Range must be between 0 and 100%"}}), 400
    if payload["ph"] < 0 or payload["ph"] > 14:
        return jsonify({"error": {"ph": "Range must be between 0 and 14"}}), 400

    df_in = pd.DataFrame([payload], columns=CROP_FEATURES)
    pred = crop_model.predict(df_in)[0]
    return jsonify({"crop": pred})


@app.route("/api/fertilizer", methods=["POST"])
def predict_fertilizer():
    expected = {
        "Temparature": float,
        "Humidity": float,
        "Moisture": float,
        "soil_type": str,
        "crop_type": str,
        "Nitrogen": int,
        "Potassium": int,
        "Phosphorous": int,
    }

    payload, errors = _extract_payload(expected)
    if errors:
        return jsonify({"error": errors}), 400

    payload["Soil Type"] = payload.pop("soil_type")
    payload["Crop Type"] = payload.pop("crop_type")

    if payload["Temparature"] < -10 or payload["Temparature"] > 60:
        return jsonify({"error": {"Temparature": "Range must be between -10 and 60 C"}}), 400
    if payload["Humidity"] < 0 or payload["Humidity"] > 100:
        return jsonify({"error": {"Humidity": "Range must be between 0 and 100%"}}), 400
    if payload["Moisture"] < 0 or payload["Moisture"] > 100:
        return jsonify({"error": {"Moisture": "Range must be between 0 and 100%"}}), 400

    df_in = pd.DataFrame([payload], columns=FERT_FEATURES)
    pred = fert_model.predict(df_in)[0]
    return jsonify({"fertilizer": pred})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
