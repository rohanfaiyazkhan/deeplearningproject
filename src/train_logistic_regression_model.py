import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from sklearn.linear_model import SGDClassifier
from joblib import dump
import numpy as np
import pandas as pd
from data.load_data import load_faces, load_features
from utils import get_labels


np.random.seed(67)
number_of_minibatches = 3

sgd_classifier_params = {"alpha": 0.001, "penalty": 'l2', "loss": "log"}


def get_binary_accuracy(preds, actual):
    num_of_correct_predictions = (preds == actual).sum()
    accuracy = num_of_correct_predictions / len(actual)
    return accuracy


def save_model(model, dest_path):
    dump(model, dest_path)

    print(f"Saved model as {dest_path}")


if __name__ == "__main__":
    features = load_features()
    faces = load_faces()

    num_of_samples = features.shape[0]
    idx = np.random.permutation(range(num_of_samples))
    cut = int(0.8 * num_of_samples)
    train_idx = idx[:cut]
    valid_idx = idx[cut:]

    num_of_training_samples = len(train_idx)
    minibatch_chunk_size = num_of_training_samples // number_of_minibatches

    ground_truth = ['pol_dat_us', 'pol_dat_ca', 'pol_dat_uk', 'pol_fb_us']

    model = SGDClassifier(**sgd_classifier_params)

    for i in range(number_of_minibatches):
        print(f"Training with minibatch: {i}")
        indexes = train_idx[minibatch_chunk_size *
                            i: minibatch_chunk_size * (i + 1)]
        X = features[indexes]
        y = get_labels(faces.iloc[indexes])
        model.partial_fit(X, y, classes=[0, 1])

    y_preds = model.predict(features[valid_idx]).astype(float)
    y_preds_proba = model.predict_proba(features[valid_idx])
    y_valid = get_labels(faces.iloc[valid_idx])

    valid_accuracy = get_binary_accuracy(y_preds, y_valid)

    y_train_pred = model.predict(features[train_idx]).astype(float)
    y_train = get_labels(faces.iloc[train_idx])

    training_accuracy = get_binary_accuracy(y_train_pred, y_train)

    print(
        f"Model trained with training accuracy: {training_accuracy} and validation accuracy: {valid_accuracy}")

    save_model(model, 'saved_model/log.joblib')
