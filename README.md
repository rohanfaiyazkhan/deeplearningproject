# Reproduction of "Facial recognition technology can expose political orientation from naturalistic facial images"


This project is aimed at reproduction and interpretation of the methods described in the paper “Facial recognition technology can expose political orientation from naturalistic facial images” [\[1\]](#kosinski_ref). Due to the ethical ramifications of the claims made in the study, it is important to attempt to verify and interpret these results. The main claim of the original paper is that facial recognition AI can detect political orientation from pictures with accuracy of upto 72\% which is claimed to be higher than human intuition (55\%) thus implying that facial recognition algorithm are much better than humans at detecting political orientation. However by reproducing the models and interpreting the results using LIME, we call into question whether the models are learning anything of note.

## Requirements

To install requirements, run:

```bash
pip install -r requirements.txt
```

## Dataset

Datasets can be downloaded from the link specified in the paper. [https://osf.io/c58d3/](https://osf.io/c58d3/)

Download the `RData` files to `/data` and run:

```python
python ./data/convert_rdata_to_numpy.py
```
This outputs a `vgg.npy` containing the features and `faces.csv` containing the labels. This data can be loaded at any time using the functions provided in `data/load_data.py`.

## Code

A copy of the original R-script provided by the author is available in `original_kosinski_code.R`.

Please note that the experiments were run in Jupyter Lab environment. We cannot guarantee that any other environment can run this without errors.

To train the models run the notebooks `train_lasso_regression.ipynb` and `train_neural_net.ipynb` respectively. `train_lasso_regression_cross_validation.ipynb` is also available as a means to reproduce Table 2 of the original paper.

Before running LIME please ensure that the pre-trained weights for ResNet-50 model using VGGFace2 dataset are available. While the original weights provided by the authors of VGGFace2 are no longer available, a backup of it can be found [here](https://queensuca-my.sharepoint.com/:u:/g/personal/21rfk_queensu_ca/EQ7o1aRxfIFNprDBi0H01sQBp7rmAKfyu-Jwjk3K3AHJWA?e=gAPoQs). Download it and copy the weights as `saved_model/resnet50_ft_weight.pkl`. Afterwards LIME can be run on any sample image using `lime_lass.ipynb` and `lime_neural_net.ipynb`.

## Reference

<span id="kosinski_ref">[\[1\] M. Kosinski, “Facial recognition technology can expose political orientation from naturalistic facial images,” Scientific Reports, vol. 11, no. 1, Art. no. 1, Jan. 2021, doi: 10.1038/s41598-020-79310-1.](https://www.nature.com/articles/s41598-020-79310-1)</span>


