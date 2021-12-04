import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple binary classifier that takes a 2048 feature long tensor as input
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()        
        
        # Number of input features is 2048
        self.layer_1 = nn.Linear(2048, 2048)
        self.layer_2 = nn.Linear(2048, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.dropout(x)
        x = self.layer_2(x)
        
        return x
    
def load_pretrained_classifier(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BinaryClassifier()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model.to(device)
    

if __name__ == '__main__':
    model = load_pretrained_classifier("./saved_model/weights-2.pth")
    print(model)
    