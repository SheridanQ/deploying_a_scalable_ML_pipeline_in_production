from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, encoder, lb):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, "..", "..", "model", "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(base_dir, "..", "..", "model", "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    with open(os.path.join(base_dir, "..", "..", "model", "lb.pkl"), "wb") as f:
        pickle.dump(lb, f)


def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, "..", "..", "model", "model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(base_dir, "..", "..", "model", "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(base_dir, "..", "..", "model", "lb.pkl"), "rb") as f:
        lb = pickle.load(f)

    return model, encoder, lb
