import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import joblib
from tqdm import tqdm
from data.load_data import load_faces, load_features
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd



features = load_features()
faces = load_faces()

np.random.seed(67)


def get_sample_and_label(row):
    for i in ground_truth:
        if ~np.isnan(row[i]):
            return i, row[i]
    print(f"No label found for userid: {row['userid']}")


def get_not_nan_labels(colname):
    col = faces[colname]
    return col[col.notna()]


def get_binary_accuracy(preds, actual):
    num_of_correct_predictions = (preds == actual).sum()
    accuracy = num_of_correct_predictions / len(actual)
    return accuracy


def create_model():
    # alpha 0.0001 penalty  seems to be optimal
    params = {
        "alpha": 0.0001, "penalty": 'l2', "loss": 'log'
    }

    return SGDClassifier(**params)


def save_model(model, filepath):
    joblib.dump(model, filepath)


ground_truth = ['pol_dat_us', 'pol_dat_ca', 'pol_dat_uk', 'pol_fb_us']

max_number_of_chunks = 4
max_chunksize = len(features) // max_number_of_chunks

if __name__ == "__main__":

    samples = {k: get_not_nan_labels(k) for k in ground_truth}

    models = {k: create_model() for k in ground_truth}

    with tqdm(ground_truth) as t:
        for sample_name in t:
            t.set_description(f'Sample {sample_name}')
            model = models[sample_name]

            # get indexes and values for each sample
            sample = samples[sample_name]
            indexes = np.array(sample.index) - 1
            values = np.array(sample.values)

            num_of_batches = len(indexes) // max_chunksize

            if num_of_batches == 0:
                y = values
                X = features[indexes]
                model.fit(X, y)
            else:
                for i in range(num_of_batches):
                    batch_idx = indexes[max_chunksize *
                                        i: max_chunksize * (i + 1)]
                    y = values[max_chunksize * i: max_chunksize * (i + 1)]
                    X = features[batch_idx]
                    model.partial_fit(X, y, classes=[0, 1])

            models[sample_name] = model

    acc_matrix = np.zeros([4, 4])

    with tqdm(enumerate(ground_truth)) as t:
        for i, sample_name in enumerate(ground_truth):
            t.set_description(f'Sample {sample_name}')
            model = models[sample_name]

            for j, test_sample in enumerate(ground_truth):
                t.set_postfix({'imputing': test_sample})

                sample = samples[test_sample]
                indexes = np.array(sample.index) - 1
                values = np.array(sample.values)

                X = features[indexes]
                y_pred = model.predict(X)

                acc = get_binary_accuracy(y_pred, values)

                acc_matrix[i, j] = acc

    df = pd.DataFrame(acc_matrix * 100, columns=[f'accuracy on {i}' for i in ground_truth], index=[
        f'model trained on {i}' for i in ground_truth])

    df.to_csv("./results/table_2_reproduced.csv")

    for sample_name in ground_truth:
        save_model(models[sample_name],
                   f'./saved_model/lasso_{sample_name}.joblib')
