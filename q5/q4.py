import torch
import torch.nn.functional as F
from torch import nn
from deeplib.history import History
import numpy as np

from deeplib.datasets import load_mnist
from deeplib.training import train
from q5_helper import get_trainable_params

import matplotlib.pyplot as plt

# Fully connected feedforward net that fits MNIST perfectly
class OverfitNet(nn.Module):

    def __init__(self, layer_sizes=[100, 100]):
        super(OverfitNet, self).__init__()

        self.input_size = 784 # 28x28
        self.num_classes = 10

        layer_sizes = [self.input_size] + layer_sizes + [self.num_classes]

        self.fc_layers = []
        for i in range(len(layer_sizes) - 1):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.add_module("fc"+str(i+1), self.fc_layers[-1])

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        # don't apply ReLU to the logit outputs
        for fc in self.fc_layers[:-1]:
            x = F.relu(fc(x))
        
        x = self.fc_layers[-1](x)
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    epochs = 30

    # large batch sizes cause overfitting, which is exactly what we want
    batch_size = 2048

    # we have only 1 large hidden layer
    hidden_sizes = [4096]

    lr = 0.006

    net = OverfitNet(layer_sizes=hidden_sizes)
    print("OverfitNet:")
    print(net)

    print(f"Epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"Learning rate: {lr}")

    optimizer = torch.optim.Adam(get_trainable_params(net), lr=lr)

    train_ds, test_ds = load_mnist('data/')
    history = train(net, optimizer, dataset=train_ds, n_epoch=epochs, batch_size=batch_size)

    # display the train/validation accuracy and loss
    history.display()
    plt.show()




    