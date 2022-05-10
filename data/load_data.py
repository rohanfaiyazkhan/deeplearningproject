import numpy as np
import pandas as pd
from pathlib import Path

base_path = Path("./data/original")


def load_features():
    return np.load(base_path / 'vgg.npy')


def load_faces():
    return pd.read_csv(base_path / 'faces.csv', index_col=0)
