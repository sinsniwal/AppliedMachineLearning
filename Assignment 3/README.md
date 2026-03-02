## Spam Scoring API

A Flask-based REST API that serves a pre-trained scikit-learn model to evaluate text messages and score their spam propensity.

### Repository Structure

- **`app.py`**: Flask application serving the scoring endpoint.
- **`score.py`**: Core inference logic. Extracts probability and applies classification thresholds.
- **`load_model.py`**: Connects to the local MLflow registry to fetch and cache the trained `SVM` model.
- **`test.py`**: Pytest suite containing unit and API integration tests.

### Running the API

Start the server:

```bash
python3 app.py
```

### Endpoint: `POST /score`

**Request:**

```json
{
  "text": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
  "threshold": 0.5
}
```

_(The `threshold` field is optional and defaults to 0.5)._

**Response:**

```json
{
  "prediction": true,
  "propensity": 0.985
}
```

### Testing & Coverage

The test suite (`test.py`) includes:

1. **Unit Tests (`test_score`)**: Validates format constraints, probability boundaries (0.0 to 1.0), prediction mapping, and threshold edge cases (0.0 vs 1.0).
2. **Integration Tests (`test_flask`)**: Automates the Flask server startup/teardown, hits the live endpoint, and verifies HTTP status codes and JSON schema integrity.

Run the tests:

```bash
pytest --cov=score test.py
```

### Coverage Report

The scoring logic achieves complete statement coverage:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.0, pytest-9.0.2, pluggy-1.5.0
rootdir: /Users/mohit/github/AppliedMachineLearning/temp/ass3
plugins: cov-7.0.0, anyio-4.12.0
collected 2 items

test.py ..                                                               [100%]

================================ tests coverage ================================
_______________ coverage: platform darwin, python 3.11.0-final-0 _______________

Name        Stmts   Miss  Cover
------------------------------
score.py        5      0   100%
------------------------------
TOTAL           5      0   100%
============================== 2 passed in 5.06s ===============================
```
