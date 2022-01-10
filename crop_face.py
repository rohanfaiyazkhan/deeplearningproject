from facenet_pytorch import MTCNN

from PIL import Image
import numpy as np

default_params = {'post_process':False, 'device':'cuda:0'}

def prepare_multi_face_model(margin=20):
    return MTCNN(margin=margin, keep_all=True, **default_params)

def prepare_single_face_model(margin=20):
    return MTCNN(margin=margin, keep_all=False, **default_params)
    
def convert_face_tensor_to_numpy_array(face_tensor):
    return face_tensor.permute(1, 2, 0).int().numpy().astype(np.uint8)

def crop_faces(img_path, keep_all=True, **kwargs):
    img = Image.open(img_path)
    if keep_all:
        mtcnn = prepare_multi_face_model(**kwargs)
    else:
        mtcnn = prepare_single_face_model(**kwargs)
    return mtcnn(img)

def crop_and_save_one_face(img_path, dest_path):
    face = crop_faces(img_path, keep_all=False)
    
    if face:
        face_arr = convert_face_tensor_to_numpy_array(face)
        Image.fromarray(face_arr).save('dest_path')
    else:
        print("No face could be extracted")
        
if __name__ == "__main__":
    crop_and_save_one_face("./sample.jpg", "./")
