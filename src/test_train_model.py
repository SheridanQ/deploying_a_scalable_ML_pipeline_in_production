import numpy as np
import pytest
import pandas as pd
from src.ml.data import load_data
from sklearn.model_selection import train_test_split

from src.ml.model import inference, compute_model_metrics, load_model
from src.train_model import categorical_features, process_data


@pytest.fixture(scope="session")
def data():
    df = load_data()
    return df


def test_null(data):
    assert data.shape == data.dropna().shape


def test_age_range(data):
    assert data['age'].between(0, 120).all()


def test_inference(data):
    """
    Test the type and value return by inference() is correct
    """
    trained_model, encoder, lb = load_model()
    _, test = train_test_split(data, test_size=0.20)

    X_test, _, _, _ = process_data(
        test,
        categorical_features=categorical_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        training=False)
    prediction = inference(trained_model, X_test)

    assert isinstance(prediction, np.ndarray)
    assert list(np.unique(prediction)) == [0, 1]


def test_compute_model_metrics(data):
    """
    Test the range of performance metrics returned by compute_model_metrics()
    """
    trained_model, encoder, lb = load_model()
    _, test = train_test_split(data, test_size=0.20)

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=categorical_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        training=False)
    prediction = inference(trained_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, prediction)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


if __name__ == "__main__":
    data = load_data()
    print(data.head())
    test_null(data)
    test_age_range(data)
    test_inference()
    test_compute_model_metrics()
