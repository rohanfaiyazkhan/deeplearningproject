

import torch
from saved_model.resnet import ResNet, Bottleneck

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resnet50(weights_path=None, **kwargs):
    """Constructs a ResNet-50 model with optional pretrained weights
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if weights_path:
        import pickle
        with open(weights_path, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        model.load_state_dict(weights)
    return model

def prepare_resnet_model(weights_path="./resnet50_ft_weight.pkl"):
    model = resnet50(weights_path, num_classes=8631)  # Pretrained weights fc layer has 8631 outputs
    model.eval()
    model.to(device)
    return model