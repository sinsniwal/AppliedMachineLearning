import mlflow
import joblib


def load_mlflow_model(uri,model_name,version=1):
    """
    Connects to the MLflow database and extracts the best model, saves it as a .pkl file.
    """
    try:
        mlflow.set_tracking_uri(uri)
        print(f"Attempting to load '{model_name}' from MLflow...")
        model_uri = f"models:/{model_name}/{version}"
        best_model = mlflow.sklearn.load_model(model_uri)
        save_path = "best_model.pkl"
        joblib.dump(best_model, save_path)

        print(f"Success! model saved to: {save_path}")

    except Exception as e:
        print(f"Failed to extract the model. Error: {e}")
        raise e


if __name__ == "__main__":
    uri= "file:///Users/mohit/github/AppliedMachineLearning/Assignment%202/mlruns"
    model_name="SVM"
    version=2
    load_mlflow_model(uri,model_name,version)  