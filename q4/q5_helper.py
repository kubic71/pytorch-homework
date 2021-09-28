import collections
import string

import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor


# Datasets
def load_fashion(path: str) -> (FashionMNIST, FashionMNIST):
    """
    Loads Fashion MNIST dataset
    :param path: path to save the dataset
    :return: (train, test)
    """
    train = FashionMNIST(path, train=True, download=True)
    test = FashionMNIST(path, train=False, download=True)
    return train, test


def prediction_to_letter(x: int) -> str:
    if x < 0 or x > 25:
        raise ValueError('Invalid letter number')
    return string.ascii_letters[x]


def _target_transform(x: int) -> int:
    return x - 1


class ReducedEMNIST(Dataset):
    def __init__(self, path: str, num_per_class: int):
        """
        EMNIST dataset reduced so that each class only has `num_per_class` samples
        :param path: path to save the dataset
        :param num_per_class: number of examples per class
        """
        self.dataset = EMNIST(path, split='letters', train=True, download=True, transform=ToTensor(),
                              target_transform=_target_transform)
        indexes_per_classes = collections.defaultdict(list)
        for i, (_, y) in enumerate(self.dataset):
            indexes_per_classes[y].append(i)
        for k in indexes_per_classes.keys():
            indexes_per_classes[k] = indexes_per_classes[k][:num_per_class]
            if len(indexes_per_classes[k]) != num_per_class:
                raise RuntimeError(f'Class {prediction_to_letter(k)} has not enough samples')
        self.kept_indexes = []
        for v in indexes_per_classes.values():
            self.kept_indexes.extend(v)

    def __len__(self):
        return len(self.kept_indexes)

    def __getitem__(self, idx):
        return self.dataset[self.kept_indexes[idx]]


def load_emnist(path: str) -> (ReducedEMNIST, EMNIST):
    """
    Loads Reduced EMNIST dataset
    :param path: path to save the dataset
    :return: (train, test)
    """
    train = ReducedEMNIST(path, 250)
    test = EMNIST(path, split='letters', train=False, download=True, transform=ToTensor(),
                  target_transform=_target_transform)
    return train, test


# Utils
def freeze_model(model) -> None:
    """
    Freeze all layers of a model
    :param model: model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False


def get_trainable_params(model):
    """
    Returns all trainable parameters (parameters where requires_grad is true)
    :param model: model
    :return: list of trainable parameters
    """
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    return params_to_update


# Network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 150, 3, padding=1)
        self.conv3 = nn.Conv2d(150, 300, 3, padding=1)
        self.conv4 = nn.Conv2d(300, 300, 3, padding=1)
        self.conv5 = nn.Conv2d(300, 150, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(150 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
