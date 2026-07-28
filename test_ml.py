import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, inference, compute_model_metrics


def test_train_model_returns_random_forest():
    """
    Test that train_model returns a trained RandomForestClassifier instance.
    """
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_inference_output_shape():
    """
    Test that inference returns predictions with the same number of rows
    as the input data.
    """
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert len(preds) == len(y_train)


def test_compute_model_metrics_values():
    """
    Test that compute_model_metrics returns precision, recall, and F1
    as floats within the valid range of 0 to 1.
    """
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0