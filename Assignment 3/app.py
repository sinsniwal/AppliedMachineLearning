from flask import Flask, request, jsonify
from load_model import load_mlflow_model
from score import score
import joblib
import os

app = Flask(__name__)


def get_model():
    MODEL_PATH = "best_model.pkl"
    return joblib.load(MODEL_PATH)


try:
    model = get_model()
except Exception as e:
    raise Exception(f"Failed to load model: {e}")


@app.route("/", methods=["GET"])
def home():
    return "<h1>SMS Spam Scoring API</h1><p>Use POST /score with JSON to get predictions.</p>"


@app.route("/score", methods=["POST"])
def score_endpoint():
    global model
    try:
        if request.content_type != "application/json":
            return jsonify({"error": "Request must be JSON"}), 415

        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 415

        # Check required fields and types
        if "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 422
        if not isinstance(data["text"], str):
            return jsonify({"error": "'text' field must be a string"}), 422

        # Check optional threshold
        threshold = data.get("threshold", 0.5)
        if not isinstance(threshold, (int, float)):
            return jsonify({"error": "'threshold' must be a number"}), 422

        text = data["text"]

        if model is None:
            model = get_model()

        pred, prop = score(text, model, float(threshold))

        return jsonify({"prediction": pred, "propensity": prop})
    except Exception as e:
        return jsonify({"error": str(e)}), 422


if __name__ == "__main__":
    uri = "file:///Users/mohit/github/AppliedMachineLearning/Assignment%202/mlruns"
    model_name = "SVM"
    version = 2
    load_mlflow_model(uri, model_name, version)

    app.run(debug=True, port=5001)
