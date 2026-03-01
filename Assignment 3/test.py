import pytest
import os
import time
import requests
import joblib
from score import score

MODEL_PATH = "best_model.pkl"

@pytest.fixture(scope="session")
def trained_model():
    """Load the model once for the entire test session."""
    if not os.path.exists(MODEL_PATH):
        from load_model import load_mlflow_model
        load_mlflow_model()
        
    model = joblib.load(MODEL_PATH)
    assert model is not None, "Setup failed: Could not load the model from disk."
    return model


def test_score(trained_model):
    """Unit test for the score function covering basic scenarios, edges, and sanity checks."""
    spam_msg = "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!"
    safe_msg = "Hey, are we still meeting for lunch today?"

    # 1. Smoke test
    pred_spam, prop_spam = score(spam_msg, trained_model, 0.5)
    pred_safe, prop_safe = score(safe_msg, trained_model, 0.5)
    
    # Check that we actually got values back
    assert pred_spam is not None and prop_spam is not None
    assert pred_safe is not None and prop_safe is not None

    # 2. Format and type checking
    assert isinstance(pred_spam, (bool, int)), f"Expected bool or int, got {type(pred_spam)}"
    assert isinstance(prop_spam, float), f"Expected float, got {type(prop_spam)}"

    # 3. Sanity checks on output bounds
    assert pred_spam in [True, False, 0, 1], f"Invalid prediction value: {pred_spam}"
    assert 0.0 <= prop_spam <= 1.0, f"Propensity {prop_spam} is out of bounds (0-1)"
    assert 0.0 <= prop_safe <= 1.0, f"Propensity {prop_safe} is out of bounds (0-1)"

    # 4. Typical inputs mapping correctly
    assert pred_spam in [True, 1], "Model failed to catch an obvious spam text."
    assert pred_safe in [False, 0], "Model incorrectly flagged a normal text as spam."

    # 5. Edge case: 0.0 threshold (everything gets flagged)
    edge_pred_0, _ = score(spam_msg, trained_model, 0.0)
    edge_pred_0_safe, _ = score(safe_msg, trained_model, 0.0)
    assert edge_pred_0 in [True, 1], "Threshold 0.0 should flag spam message as True"
    assert edge_pred_0_safe in [True, 1], "Threshold 0.0 should flag safe message as True"

    # 6. Edge case: 1.0 threshold (nothing gets flagged)
    edge_pred_1, _ = score(spam_msg, trained_model, 1.0)
    assert edge_pred_1 in [False, 0], "Threshold 1.0 should result in False prediction"


def test_flask(trained_model):
    """Integration test to verify the Flask endpoint."""
    # Boot up the app via CLI in the background
    os.system("python3 app.py &")
    
    # Give the server time to initialize
    time.sleep(3)

    try:
        payload = {"text": "URGENT! You have won a 1 week FREE membership"}
        resp = requests.post("http://127.0.0.1:5001/score", json=payload)
        
        # Network & HTTP checks
        assert resp is not None, "Server did not return a response."
        assert resp.status_code == 200, f"Expected HTTP 200 OK, but got {resp.status_code}"
        assert "application/json" in resp.headers.get("Content-Type", ""), "Endpoint did not return JSON"
        
        data = resp.json()
        
        # Schema validation
        assert "prediction" in data, "Response payload missing 'prediction' key"
        assert "propensity" in data, "Response payload missing 'propensity' key"
        
        # Type validation on JSON data
        assert isinstance(data["prediction"], (bool, int)), "JSON prediction is not a boolean"
        assert isinstance(data["propensity"], float), "JSON propensity is not a float"

        # Accuracy check: Does the API match the local python function?
        local_pred, local_prop = score(payload["text"], trained_model, 0.5)
        assert data["prediction"] == local_pred, "API prediction does not match local function prediction"
        assert abs(data["propensity"] - local_prop) < 1e-5, "API propensity score mismatch"

    finally:
        # Teardown the app safely
        os.system("pkill -f 'python3 app.py'")