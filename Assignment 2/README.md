**Project**: SMS Spam Classification (MLOps Pipeline)

**Pipeline**: Implemented data versioning (DVC) and experiment tracking (MLflow). Standardized data splits were used to ensure comparable results. Models tested include Logistic Regression, Random Forest, and SVM.

**Results**:
Logistic Regression: 0.9739 AUCPR
Random Forest: 0.9787 AUCPR
SVM: 0.9807 AUCPR

**Champion Model**:
The best model was SVM with an AUCPR of 0.9807.
Parameters: Kernel='rbf', C=2.0.

**Notes**:

- Run generated identical results for both data versions due to consistent performance across seeds.
- To execute: Run `prepare.ipynb` then `train.ipynb`.
- View detailed logs: `mlflow ui`
