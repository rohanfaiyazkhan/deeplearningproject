import numpy as np
import pandas as pd

def load_features():
    return np.load('./data/vgg.npy')

def load_faces():
    return pd.read_csv('data/faces.csv', index_col=0)