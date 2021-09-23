# Fully connected feedforward net that fits MNIST perfectly
import torch
import torch.nn.functional as F
from torch import nn
from deeplib.history import History
import poutyne
import numpy as np
import math

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

#class DatasetSubset(t):
#    def __init__(self, data, labels, n=100):
#        self.n = n
#        self.data = data
#        self.labels = labels
#
#    def __len__(self):
#        return self.n
#
#    def __getitem__(self, idx):
#        self.data[idx], self.labels[idx]
#
#        img = Image.fromarray(img.numpy(), mode='L')
#
#        if self.transform is not None:
#            img = self.transform(img)
#
#        if self.target_transform is not None:
#            target = self.target_transform(target)


from poutyne import one_cycle_phases
from poutyne import OptimizerPolicy


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    epochs = 600
    batch_size = 4096

    # initial lr, max lr, final lr
    lr_scale = 1
    lr = tuple(map(lambda l: l*lr_scale,  (0.001, 0.01, 0.0003)))

    # 80% of the MNIST is the train set
    steps_per_epoch = math.ceil((60000 * 0.8) / batch_size)
    # print("Steps per epoch:", steps_per_epoch)

    train_ds, test_ds = load_mnist('data/')
    net = OverfitNet(layer_sizes=[400])
    net_params = get_trainable_params(net)
    print(net)
    # optimizer = torch.optim.Adam(get_trainable_params(net), lr=0.01)
    optimizer = torch.optim.Adam(net_params, lr=lr[0])
    policy = OptimizerPolicy(
        one_cycle_phases(epochs * steps_per_epoch, lr=lr),
    )

    grad_clip = poutyne.ClipNorm(net_params, max_norm=5)

    # policy = poutyne.StepLR(step_size=15, gamma=0.1) 
    train(net, optimizer, dataset=train_ds, n_epoch=epochs, batch_size=batch_size, callbacks=[policy, grad_clip])


    