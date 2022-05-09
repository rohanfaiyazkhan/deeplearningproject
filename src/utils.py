import numpy as np
import cv2
import os
from pathlib import Path
from PIL import Image

ground_truth = ['pol_dat_us', 'pol_dat_ca', 'pol_dat_uk', 'pol_fb_us']
image_file_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

def label_func(row):
    '''
    Checks each column of ground_truth and extracts whichever column is not null as the label.
    0 represents liberal and 1 represents conservative
    '''
    for i in ground_truth:
        if ~np.isnan(row[i]):
            return row[i]
    return np.nan


def is_image_path_valid(path: Path):
    return path.is_file() and path.suffix in image_file_extensions

def verify_image(fn):
    "Confirm that `fn` can be opened"
    try:
        im = Image.open(fn)
        im.draft(im.mode, (32,32))
        im.load()
        return True
    except: return False

def load_image(path):
    return Image.open(path)

def get_labels(data):
    '''
    Returns array of labels from entire dataset
    '''
    return data.apply(lambda row: label_func(row), axis=1).to_numpy()

def load_images_recursively(root_dir: Path):
    ls = os.listdir
    
    images = []
    label2image = []
    
    def append_if_image(root: Path, filename: str):
        path = root / filename
        
        if is_image_path_valid(path):
            images.append(path)
            label2image.append(root.stem)
        
    for filename in ls(root_dir):
        file_path = root_dir / filename
            
        if file_path.is_dir():
            for nested_filename in ls(file_path):
                append_if_image(file_path, nested_filename)
        else:
            append_if_image(root_dir, filename)
            
    return images, label2image