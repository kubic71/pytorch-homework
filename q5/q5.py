import numpy as np
import torch

from deeplib.datasets import load_mnist
from q5_helper import load_emnist, Network, load_fashion

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    # Training parameters
    # TODO insert training parameters such as n_epoch, batch_size and lr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset
    train_set, test_set = load_mnist('data/')
    train_set, test_set = load_fashion('data/')
    train_set, test_set = load_emnist('data/')

    # Model
    model = Network().to(device)

    # Load weights from file
    # model.load_state_dict(torch.load('path/to/weights.pth'))

    # Freeze model
    # freeze_model(model)

    # Modify the last layer of the network
    # model.fc3 = nn.Linear(model.fc2.out_features, 10)

    # Training
    # TODO insert training code (you can use the train and test functions from deeplib)

    # Saving model
    # pretrained_folder = pathlib.Path('pretrained')
    # pretrained_folder.mkdir(exist_ok=True)
    # save_path = pretrained_folder.joinpath('state_name.pth')
    # torch.save(model.state_dict(), str(save_path))
