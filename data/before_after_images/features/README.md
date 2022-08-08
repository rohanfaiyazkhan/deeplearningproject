# Extracted features

This directory contains extracted features from the dataset for both E2 and E3 from the paper. The purpose of this is to make the dataset public without compromising privacy and confidentiality for those pictured in the images.

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
