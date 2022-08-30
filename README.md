# Codebase associated with paper: "Before And After Pseudoscience: An Empirical Challenge to Social Classification using Facial Recognition"

## Abstract

A number of researchers and companies claim to be able to discern social category membership (criminality, sexual orientation, political orientation, etc.) using facial recognition tools, and suggest on that basis that these categories have a biological basis that can be detected with automated tools. Possible use cases suggested include policing applications, and government surveillance. Critics have raised ethical, political and methodological challenges to these claims. Here we conduct an empirical investigation of a study that claims to be able to discern political orientation from images of faces posted to social media, with the aim of teasing out how much of the predictive power of the model can be attributed to grooming and presentation choices rather than deeper biological characteristics. We create a novel dataset consisting of pairs of images of where one individual is presenting themselves differently in the two images. These pairs of images are drawn from searches for “before and after” images from makeovers, haircuts, and drag. We re-implement the political orientation detection model and use it to classify the images from our dataset, to determine test-retest reliability. Individuals' predicted political orientations differ for the two images 33% of the time. We also retrain the models to predict “before” and “after” categories on the new dataset, and achieve prediction accuracy of 68%, which is comparable to the political orientation model's results. These findings suggest that superficial presentation differences, not biological factors, are responsible for comparable categorization results.

## Requirements

To install requirements, run:

```bash
pip install -r requirements.txt
```

## Dataset

Datasets can from logistic regression and neural network models are the same as the original paper being re-implemented [\[1\]](#ref) and can be downloaded from this link. [https://osf.io/c58d3/](https://osf.io/c58d3/)

Download the `RData` files to `/data/original` and run:

```python
python ./data/convert_rdata_to_numpy.py
```

This outputs a `vgg.npy` containing the features and `faces.csv` containing the labels. This data can be loaded at any time using the functions provided in `data/load_data.py`.

Unfortunately the raw dataset for "Before and After" dataset cannot be made public to preserve anonymity and privacy of persons pictured. Unfortunately this limits the reproducibility of the project. We have however made public the exact scripts used to scrape the images (available [here](https://anonymous.4open.science/r/bing-search-image-scraping-EFB1)) and the search terms used to generate images detailed in the paper. We also made available a tool for labelling images more efficiently [here](https://anonymous.4open.science/r/tkinter-image-labeller-gui-01F3/). The data should be saved in the following structure (the directories are already committed in the repository):

```
data
└── before_after_images
    └── processed
        ├── after
        └── before
```

However to allow for some level reproducibility we have made available the list of extracted features from the raw images using both ResNet-50 trained on VGGFace2 (i.e. for test-retest analysis in E2) and ResNet-18 trained on ImageNet (i.e. for Before-After model in E3). The features are available in `data/before_after_images/features`. Please see `data/before_after_images/features/README.md` for more details.

If you have the dataset regenerated and can proceed with test-retest analysis, please ensure that the pre-trained weights for ResNet-50 model using VGGFace2 dataset are available. While the original weights provided by the authors of VGGFace2 are no longer available, a backup of it can be found [here](https://queensuca-my.sharepoint.com/:u:/g/personal/21rfk_queensu_ca/EQ7o1aRxfIFNprDBi0H01sQBp7rmAKfyu-Jwjk3K3AHJWA?e=gAPoQs). Download it and copy the weights as `saved_model/resnet50_ft_weight.pkl`.

## Models

Originally, models used to report results in the paper were trained in the notebooks `notebooks/train_lasso_regression.ipynb` and `notebooks/train_neural_net.ipynb` to train the logistic regression model and neural network respectively. The training of the Before-After model can also be found in `notebooks/before_and_after_model.ipynb`.

The relevant parts have been cleaned into python scripts found in the `src` directory for easier reproduction.

## Reproducing Experiments with Extracted Features

The directory `data/before_after_images/features` contains extracted features from the dataset for both E2 and E3 from the paper. The purpose of this is to make the dataset public without compromising privacy and confidentiality for those pictured in the images.

The structure of the data is as follows:

```
features
└── resnet_18_imagenet
    ├── after
    |   └── list_of_tensors.pt
    └── before
        └── list_of_tensors.pt
└── resnet_50_vggface2
    ├── after
    |   └── list_of_tensors.pt
    └── before
        └── list_of_tensors.pt
```

The data can be loaded to perform additional training (similar to the models used in E2) or for inference. The data should be loaded directly as tensors using `torch.load`. For example:

```python
list_of_tensors = torch.load("data/before_after_images/features/resnet_18_imagenet/before/list_of_tensors.pt")
```

Inference can then be performed afterwards as follows:

```python
learn_inf = load_learner("../saved_model/before_after.pkl")
sm = torch.nn.Softmax()
last_layer = learn_inf.model[-1][-1]
res = sm(last_layer(list_of_tensors))
label = "after" if res[0][0] > 0.5 else "before"
```

Scripts for reproducing experiments with extracted features are available under `src/reproduction_extracted_features`. Either run:

```bash
# E2
python3 ./src/reproduction_extracted_features/e2.py

# E3
python3 ./src/reproduction_extracted_features/e3.py
```
