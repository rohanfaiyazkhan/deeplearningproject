
import math
from pathlib import Path
from fastai import *
from fastai.vision.all import *
from fastai.data.all import *
from pathlib import Path
from PIL import Image
import math
import numpy as np
import torch
from utils import get_torch_device


def predict(fn, model, scale=1):
    '''
    Handy function for getting prediction from an image
    '''
    sample_dir = Path("./sample_images")
    before = sample_dir / fn

    img = Image.open(before)
    img = img.resize(
        (math.floor(img.shape[1] * scale), math.floor(img.shape[0] * scale)))
    return model.predict(before)


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_torch_device()

    data_path = Path('./data/before_after_images/processed')

    fns = get_image_files(data_path)

    failed = verify_images(fns)

    db = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=seed),
        get_y=parent_label,
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms())

    dls = db.dataloaders(data_path)

    learn = cnn_learner(dls, resnet18, metrics=accuracy)

    lr = 3e-3  # found using learn.lr_find()
    learn = cnn_learner(dls, resnet18, metrics=accuracy, lr=lr)
    learn.fine_tune(2, freeze_epochs=6)

    train_loss, train_accuracy = learn.validate(dl=dls.train)
    train_accuracy

    interp = ClassificationInterpretation.from_learner(learn)

    learn.model_dir = "saved_model"
    dest_path = "./saved_model/before_after.pkl"
    learn.export(dest_path)
    print(f"Model trained and saved to {dest_path}")

    # learn_inf = load_learner("saved_model/before_after.pkl")
