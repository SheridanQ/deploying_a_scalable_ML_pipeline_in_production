# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import os


# Add the necessary imports for the starter code.
from ml.data import process_data, load_data
from ml.model import (train_model,
                      compute_model_metrics,
                      inference,
                      save_model)
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Add code to load in the data.
logging.info('Loading data')
data = load_data()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Proces the test data with the process_data function.
logging.info("Preprocessing data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=categorical_features, label="salary", training=True
)
logging.info("Preprocessing test data")
X_test, y_test, _, _ = process_data(
    test, categorical_features=categorical_features, label="salary",
    training=False, encoder=encoder, lb=lb)

# Train and save a model.


def main(args):
    logging.info("Train model")
    model = train_model(X_train, y_train)
    save_model(model, encoder, lb)

    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    logging.info("Printing Performance Metrics")
    print("Overall Performance Metrics of the Trained Model:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"FBeta: {fbeta:.3f}")


def slice_feature_data(model, encoder, lb, feature):
    dataframe = pd.DataFrame(
        columns=["feature", "value", "precision", "recall", "fbeta_score"])
    for value in data[feature].unique():
        slice_data = data[data[feature] == value]
        X_test, y_test, encoder, lb = process_data(
            slice_data,
            categorical_features=categorical_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )
        y_pred = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        dataframe = dataframe.append({
            "feature": feature,
            "value": value,
            "precision": precision,
            "recall": recall,
            "fbeta_score": fbeta}, ignore_index=True)

    dataframe.to_csv(os.path.join(
        "data", "slice_performance_outputs.csv"), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature for slice data.')
    parser.add_argument('--feature', type=str,
                        help='Feature to slice.', default='education')
    args = parser.parse_args()

    main(args)
