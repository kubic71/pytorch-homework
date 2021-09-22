# Fully connected feedforward net that fits MNIST perfectly
import torch
import torch.nn.functional as F
from torch import nn
from deeplib.history import History
import numpy as np

from deeplib.datasets import load_mnist
from deeplib.training import train
from q5_helper import get_trainable_params

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
        
        x = F.relu(self.fc_layers[-1](x))
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_ds, test_ds = load_mnist('data/')
    print(train_ds)

    net = OverfitNet(layer_sizes=[100, 100])

    print(net)

    optimizer = torch.optim.Adam(get_trainable_params(net), lr=0.003)
    train(net, optimizer, dataset=train_ds, n_epoch=10, batch_size=32)


    