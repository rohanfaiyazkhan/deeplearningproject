from facenet_pytorch import MTCNN

from PIL import Image
import numpy as np

def prepare_multi_face_model():
    return MTCNN(margin=20, keep_all=True, post_process=False, device='cuda:0')

def prepare_single_face_model():
    return MTCNN(margin=20, keep_all=False, post_process=False, device='cuda:0')
    
def convert_face_tensor_to_numpy_array(face_tensor):
    return face_tensor.permute(1, 2, 0).int().numpy().astype(np.uint8)

def crop_faces(img_path, keep_all=True):
    img = Image.open(img_path)
    if keep_all:
        mtcnn = prepare_multi_face_model()
    else:
        mtcnn = prepare_single_face_model()
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
