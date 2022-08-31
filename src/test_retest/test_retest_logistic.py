
from src.utils import get_torch_device
import joblib
from PIL import Image
from fastai.data.all import get_image_files
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from saved_model.prepare_resnet50 import prepare_resnet_model
from data.load_image import load_image_for_feature_extraction
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


image_file_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
image_dir = Path("../../data/before_after_images/processed")
before_dir = image_dir / 'before'
after_dir = image_dir / 'after'
log_model_path = "../../saved_model/lasso_pol_dat_us.joblib"


def is_image_path_valid(path: Path):
    return path.is_file() and path.suffix in image_file_extensions


def load_image_file(path):
    return Image.open(path)


def preprocess(path):
    return np.expand_dims(load_image_for_feature_extraction(path), 0)[0]


def load_image_paths(path: Path):
    fns = get_image_files(path)

    return fns


def full_pipeline(x, resnet_model=None, log_model=None):
    if resnet_model == None:
        resnet_model = prepare_resnet_model(
            "./saved_model/resnet50_ft_weight.pkl")

    if log_model == None:
        log_model = joblib.load(log_model_path)

    x = torch.Tensor(x.transpose(0, 3, 1, 2))  # nx3x224x224
    x = x.to(device)
    x = resnet_model(x).detach().cpu().numpy()
    return log_model.predict(x)


def get_blank_df():
    df = pd.DataFrame(index=cat_names)
    df['total'] = np.nan
    df['before liberal'] = np.nan
    df['before conservative'] = np.nan
    df['after liberal'] = np.nan
    df['after conservative'] = np.nan

    return df


def get_contingency_table(df: pd.DataFrame = None, pred_results=None, before_colname="", after_colname=""):

    if df == None:
        df = pd.DataFrame(index=cat_names)

    for cat in cat_names:
        pred_slice = pred_results[categories == cat]
        total = len(pred_slice)
        blal = len(pred_slice[(pred_slice[before_colname] == 0) & (
            pred_slice[after_colname] == 0)])
        blac = len(pred_slice[(pred_slice[before_colname] == 0) & (
            pred_slice[after_colname] == 1)])
        bcac = len(pred_slice[(pred_slice[before_colname] == 1) & (
            pred_slice[after_colname] == 1)])
        bcal = len(pred_slice[(pred_slice[before_colname] == 1) & (
            pred_slice[after_colname] == 0)])

        df.loc[[cat], ['total']] = total
        df.loc[[cat], ['blal']] = blal
        df.loc[[cat], ['blac']] = blac
        df.loc[[cat], ['bcac']] = bcac
        df.loc[[cat], ['bcal']] = bcal

    total = df.sum()
    total.name = 'total'
    df = df.append(total.transpose())

    return df


def get_cat(stem):
    cat_dict = {
        'makeupe': 'makeup',
        'hiardoo': 'hairdoo',
        'hairdoocut': 'haircut'
    }

    first_word = stem.split()[0]

    if first_word in cat_dict.keys():
        return cat_dict[stem]
    else:
        return first_word


if __name__ == "__main__":
    before_paths = load_image_paths(before_dir)
    after_paths = load_image_paths(after_dir)

    device = get_torch_device()

    results = pd.DataFrame({'sample_paths': before_paths})

    results['results_1'] = np.nan
    results['results_2'] = np.nan

    for i, (before_path, after_path) in tqdm(enumerate(zip(before_paths, after_paths)), total=len(before_paths)):
        if before_path.stem != after_path.stem:
            print(
                f"Before and after don't match for index {i}, before: {before_path}, after: {after_path}")
            break

        results.loc[i, 'results_1'] = full_pipeline(preprocess(before_path))
        results.loc[i, 'results_2'] = full_pipeline(preprocess(after_path))

    fns = results['sample_paths']

    # extract filename without extension from path
    fns = fns.map(lambda fn: Path(fn).stem)

    categories = fns.map(lambda fn: fn.split()[0])

    categories = categories.map(get_cat)

    idx = np.where(categories == 'images')[0]

    categories[idx] = fns[idx].map(lambda s: s.split()[2])

    cat_names = categories.unique()

    log_contingency_df = get_contingency_table(
        pred_results=results, before_colname="results_1", after_colname="results_2", category_names=cat_names, categories=categories)

    log_contingency_df.to_csv("./results/log_Contingency.csv")
