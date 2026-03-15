import sklearn.base

def score(
    text: str, model: sklearn.base.BaseEstimator, threshold: float
) -> tuple[bool, float]:
    """
    Scores an input text.
    """
    # Get the probability of the positive class (spam)
    propensity = float(model.predict_proba([text])[0][1])
    
    # Determine prediction based on the threshold
    prediction = bool(propensity >= threshold)

    return prediction, propensity