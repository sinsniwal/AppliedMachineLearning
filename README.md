**Repository: Applied Machine Learning**
**Dataset / Theme**: SMS Spam Classification

**Assignment 1: Baseline Machine Learning Approach**
Evaluated TF-IDF with NB, LR, and SVM models for accuracy and performance. SVM was the best performer with an Accuracy of 98%.
To run: `python prepare.ipynb` then `python train.ipynb`

**Assignment 2: MLOps Pipeline with Versioning and Tracking**
Implemented data versioning (DVC) and experiment tracking (MLflow) into the pipeline. SVM remained the champion model with an AUCPR of 0.9807.
To run: `prepare.ipynb` then `train.ipynb`
View logs with: `mlflow ui`

**Assignment 3: Flask REST API Deployment**
Created a local Flask REST API that serves the pre-trained SVM model for inference. Added complete pytest unit testing suites.
To run server: `python3 app.py`
To run tests: `pytest --cov=score test.py`

**Assignment 4: Docker Containerization and CI Hooks**
Containerized the API into a Docker image, integrated containerized API tests, and added a pre-commit Git hook to enforce passing tests.
To build container: `docker build -t spam_classifier_app .`
To run container: `docker run -d -p 5001:5001 --name spam_app spam_classifier_app`
To test locally: `pytest test.py`
