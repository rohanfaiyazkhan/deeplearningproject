
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from PIL import Image
from pathlib import Path
from pathlib import Path
import numpy as np
import pandas as pd
from fastai.data.all import get_image_files
from fastai.data.all import get_image_files

from data.load_image import load_image_for_feature_extraction

image_file_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')


def is_image_path_valid(path: Path):
    return path.is_file() and path.suffix in image_file_extensions


def load_image_file(path):
    return Image.open(path)


def load_image_paths(path: Path):
    fns = get_image_files(path)

    return fns


def get_contingency_table(df: pd.DataFrame = None, pred_results=None, before_colname="", after_colname="", categories=None, category_names=None):

    if df == None:
        df = pd.DataFrame(index=category_names)

    for cat in category_names:
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


def preprocess(path):
    return np.expand_dims(load_image_for_feature_extraction(path)[0], 0)


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
