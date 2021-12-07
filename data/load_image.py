from PIL import Image
import numpy as np

mean_rgb = (131.0912, 103.8827, 91.4953)

def load_image_for_feature_extraction(path='', shape=None):
    '''
    Referenced from VGGFace2 Paper:
    Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman, “VGGFace2: A dataset for recognising faces across pose and age,” arXiv:1710.08092 [cs], May 2018
    '''
    short_size = 224.0
    crop_size = shape
    img = Image.open(path)
    im_shape = np.array(img.size)    # in the format of (width, height, *)
    img = img.convert('RGB')

    ratio = float(short_size) / np.min(im_shape)
    img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),   # width
                           int(np.ceil(im_shape[1] * ratio))),  # height
                     resample=Image.BILINEAR)

    x = np.array(img)  # image has been transposed into (height, width)
    newshape = x.shape[:2]
    h_start = (newshape[0] - crop_size[0])//2
    w_start = (newshape[1] - crop_size[1])//2
    x = x[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
    
    # normalize colors to prevent overfitting on color differences 
    x = x - mean_rgb
    
    # returns transformed image, and original image
    return x, img