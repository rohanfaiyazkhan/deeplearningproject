import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import numpy as np
import pandas as pd

def convert_features_to_numpy(input_file="./data/original/vgg.RData", dest_file="./data/original/vgg.npy"):
    ro.r['load'](input_file)
    vgg=ro.r['vgg']
    np.save(dest_file, vgg)
    

def convert_faces_to_csv(input_file="./data/original/faces.RData", dest_file="./data/original/faces.csv"):
    ro.r['load'](input_file)
    faces=ro.r['d']
    faces.to_csv(dest_file)
    
if __name__ == "__main__":
    convert_features_to_numpy()
    convert_faces_to_csv()

