
from matplotlib import pyplot as plt
import torch
from utils import label_func, get_labels
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

from data.load_data import load_faces, load_features

np.random.seed(67)
torch.manual_seed(67)


def get_not_nan_labels(colname):
    col = faces[colname]
    return col[col.notna()]

# Check if GPU is available


def check_if_gpu_is_available():
    print(torch.cuda.is_available())


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


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, indexes, label_func=label_func):
        self.indexes, self.label_func = indexes, label_func

    def __getitem__(self, i):
        index = self.indexes[i]

        sample = torch.tensor(features[index]).float()
        label = label_func(faces.iloc[index])

        return sample, label

    def __len__(self): return len(self.indexes)


def binary_acc(y_pred, y_test):
    # Transform outputs to 0 and 1
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    # Calculate percentage of correct predictions
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]

    return acc


def plot_losses_and_metrics(losses, val_losses, accuracies, val_accuracies):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

    axes[0].plot(losses)
    axes[0].plot(val_losses)
    axes[0].set_title('model loss')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'test'], loc='upper left')

    axes[1].plot(accuracies)
    axes[1].plot(val_accuracies)
    axes[1].set_title('binary accuracy')
    axes[1].set_ylabel('acc')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['train', 'test'], loc='upper left')

    fig.tight_layout()


def get_data_sets(valid_set=1):
    sample_to_exclude = ground_truth[valid_set]

    training_sample_names = [
        name for name in ground_truth if ground_truth != valid_set]

    def get_idx(sample):
        return np.array(sample.index) - 1

    def get_y(sample):
        return np.array(sample.values)

    train_idx = np.concatenate([get_idx(samples[n])
                                for n in training_sample_names])
    train_y = np.concatenate([get_y(samples[n])
                              for n in training_sample_names])
    valid_idx = get_idx(samples[sample_to_exclude])
    valid_y = get_y(samples[sample_to_exclude])

    return train_idx, train_y, valid_idx, valid_y


if __name__ == "__main__":
    features = load_features()
    faces = load_faces()

    # Specify that we want to use the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ground_truth = ['pol_dat_us', 'pol_dat_ca', 'pol_dat_uk', 'pol_fb_us']

    samples = {k: get_not_nan_labels(k) for k in ground_truth}

    num_of_samples = features.shape[0]

    valid_sample_name = ground_truth[1]
    valid_sample = samples[valid_sample_name]

    valid_idx = np.array(valid_sample.index) - 1
    valid_y = np.array(valid_sample.values)

    train_idx, train_y, valid_idx, valid_y = get_data_sets(0)

    train_ds = CustomDataset(train_idx)
    valid_ds = CustomDataset(valid_idx)

    train_dl = DataLoader(train_ds, batch_size=64, num_workers=6)
    valid_dl = DataLoader(valid_ds, batch_size=64, num_workers=6)

    LEARNING_RATE = 1e-4

    model = BinaryClassifier()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    EPOCHS = 10

    losses = []
    val_losses = []
    accuracies = []
    val_accuracies = []

    # Move model to GPU if possible
    model = model.to(device)
    # Tells PyTorch we are in training mode
    model.train()

    try:
        for e in range(EPOCHS):

            # Set loss and accuracy to zero at start of each epoch
            epoch_training_loss = 0
            epoch_training_accuracy = 0
            epoch_valid_loss = 0
            epoch_valid_accuracy = 0

            with tqdm(train_dl, unit="training", total=len(train_dl)) as tepoch:
                for x_batch, y_batch in tepoch:
                    tepoch.set_description(f"Epoch {e}")
                    # Transfer the tensors to the GPU if possible
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                    # Zero out gradients before backpropagation (PyTorch cumulates the gradient otherwise)
                    optimizer.zero_grad()

                    # Predict a minibatch of outputs
                    y_pred = model(x_batch)

                    # Calculate the loss (unsqueeze adds a dimension to y)
                    loss = loss_function(y_pred, y_batch.unsqueeze(1))
                    training_acc = binary_acc(y_pred, y_batch.unsqueeze(1))

                    # Backpropagation. Gradients are calculated
                    loss.backward()
                    optimizer.step()

                    batch_loss = loss.item()
                    batch_acc = training_acc.item()
                    epoch_training_loss += batch_loss
                    epoch_training_accuracy += batch_acc
                    losses.append(batch_loss)
                    accuracies.append(batch_acc)

                    # tepoch.set_postfix(loss=loss.item(), accuracy=100. * training_acc.item())

            with tqdm(valid_dl, unit="training", total=len(valid_dl)) as tepoch:
                for x_batch, y_batch in tepoch:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                    valid_y_pred = model(x_batch)
                    valid_loss = loss_function(
                        valid_y_pred, y_batch.unsqueeze(1))
                    valid_acc = binary_acc(valid_y_pred, y_batch.unsqueeze(1))

                    batch_valid_loss = valid_loss.item()
                    batch_valid_accuracy = valid_acc.item()
                    epoch_valid_loss += batch_valid_loss
                    epoch_valid_accuracy += batch_valid_accuracy
                    val_losses.append(batch_valid_loss)
                    val_accuracies.append(batch_valid_accuracy)

            avg_train_loss = epoch_training_loss/len(train_dl)
            avg_valid_loss = epoch_training_loss/len(valid_dl)

            avg_train_accuracy = epoch_training_accuracy/len(train_dl)
            avg_valid_accuracy = epoch_valid_accuracy/len(valid_dl)

            print(f'End of Epoch {e}: | Training Loss: {avg_train_loss:.5f} | Training accuracy: {avg_train_accuracy} | Validation Loss: {avg_valid_loss} | Validation Accuracy: {avg_valid_accuracy}')

    except Exception as e:
        print("Something went wrong in training")
        print(e)
        quit()

    y_pred_list = np.array([])

    model.eval()

    with torch.no_grad():
        for X_batch, y_batch in valid_dl:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_tag = y_pred_tag.squeeze(1).cpu().numpy()
            y_pred_list = np.append(y_pred_list, y_pred_tag)

    dest_path = './saved_model/weights-2.pth'
    torch.save(model.state_dict(), dest_path)
    print(f"Model saved as {dest_path}")
