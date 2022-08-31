import torch
import pandas as pd
import numpy as np
import joblib
import tqdm
from saved_model.binary_classifier import load_pretrained_classifier
import sys
from os import path

from utils import get_torch_device
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

before_tensor_path = "../data/before_after_images/features/resnet_50_vggface2/before/list_of_tensors.pt"
after_tensor_path = "../data/before_after_images/features/resnet_50_vggface2/after/list_of_tensors.pt"
log_model_path = "../../saved_model/lasso_pol_dat_us.joblib"


def log_pipeline(x, log_model=None):
    if log_model == None:
        log_model = joblib.load(log_model_path)

    return log_model.predict(x)


def nn_pipeline(x, binary_classifier=None):

    if binary_classifier == None:
        binary_classifier = load_pretrained_classifier(
            '../../saved_model/weights-2.pth')

    x = torch.sigmoid(binary_classifier(x))
    x = torch.round(x)
    return x.detach().cpu().numpy()


def get_contingency_table(df: pd.DataFrame = None, pred_results=None, before_colname="", after_colname=""):

    if df == None:
        df = pd.DataFrame()

    total = len(pred_results)
    blal = len(pred_results[(pred_results[before_colname] == 0) & (
        pred_results[after_colname] == 0)])
    blac = len(pred_results[(pred_results[before_colname] == 0) & (
        pred_results[after_colname] == 1)])
    bcac = len(pred_results[(pred_results[before_colname] == 1) & (
        pred_results[after_colname] == 1)])
    bcal = len(pred_results[(pred_results[before_colname] == 1) & (
        pred_results[after_colname] == 0)])

    df['total'] = total
    df['blal'] = blal
    df['blac'] = blac
    df['bcac'] = bcac
    df['bcal'] = bcal

    return df


def main():
    before_tensors = torch.load(before_tensor_path)
    after_tensors = torch.load(after_tensor_path)

    device = get_torch_device()

    results_log = pd.DataFrame()
    results_log['results_before'] = np.nan
    results_log['results_after'] = np.nan

    results_nn = pd.DataFrame()
    results_nn['results_before'] = np.nan
    results_nn['results_after'] = np.nan

    for i, (before_tensor, after_tensor) in tqdm(enumerate(zip(before_tensors, after_tensors)), total=len(before_tensors)):

        results_log.loc[i, 'results_before'] = log_pipeline(before_tensor)
        results_log.loc[i, 'results_after'] = log_pipeline(after_tensor)
        results_nn.loc[i, 'results_before'] = nn_pipeline(before_tensor)
        results_nn.loc[i, 'results_after'] = nn_pipeline(after_tensor)

    log_contingency_df = get_contingency_table(
        pred_results=results_log, before_colname="results_before", after_colname="results_after")

    log_contingency_df.to_csv("./log_contingency.csv")

    nn_contingency_df = get_contingency_table(
        pred_results=results_nn, before_colname="results_before", after_colname="results_after")

    nn_contingency_df.to_csv("./nn_contingency.csv")


if __name__ == "__main__":
    main()
