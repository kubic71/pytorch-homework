import numpy as np
import torch
import matplotlib.pyplot as plt

import pathlib
from deeplib.datasets import load_mnist
from deeplib.training import train, test
from q5_helper import load_emnist, Network, load_fashion, get_trainable_params

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int, help='Pretraining batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Pretraining learning rate')
    parser.add_argument('--n_epoch', default=20, type=int, help='Number of pretraining epochs')
    parser.add_argument('--dataset', type=str, required=True, choices=["mnist", "fashion"], help='Pretraining dataset - "mnist" or "fashion"')

    args = parser.parse_args()

    # pretrain MNIST or FASHION-MNIST classifiers to be used for EMNIST transfer learning
    torch.manual_seed(42)
    np.random.seed(42)

    # Training parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset
    train_set, test_set = {"mnist": load_mnist, "fashion": load_fashion}[args.dataset]("data/")

    # Model
    model = Network().to(device)
    print(model)

    print("\n\nPretraining on ", args.dataset)
    print(f"epochs: {args.n_epoch}")
    print(f"batch size: {args.batch_size}")
    print(f"learning rate: {args.lr}")

    train_params = f"n_epoch={args.n_epoch},bs={args.batch_size},lr={args.lr}"
    exp_name = f"{args.dataset}_pretraining_{train_params}"
    print(f"Experiment name: {exp_name}")

    optimizer = torch.optim.Adam(get_trainable_params(model), lr=args.lr)
    history = train(model, optimizer, dataset=train_set, n_epoch=args.n_epoch, batch_size=args.batch_size)

    train_acc = history.history["acc"][-1]
    print(f"Train accuracy: {train_acc}")

    # display the train/validation accuracy and loss
    history.display()
    plt.tight_layout()
    
    
    res_folder = pathlib.Path('results')
    res_folder.mkdir(exist_ok=True)
    plt.savefig(f"results/{exp_name}.png")

    test_acc = test(model, test_dataset=test_set, batch_size=args.batch_size)

    print(f"Test accuracy: {test_acc}")

    with open(f"results/{exp_name}_metrics.txt", "w") as f:
        f.write(f"Train accuracy: {train_acc}\n")
        f.write(f"Test accuracy: {test_acc}\n")


    # save pretrained model
    pretrained_folder = pathlib.Path('pretrained')
    pretrained_folder.mkdir(exist_ok=True)
    save_path = pretrained_folder.joinpath(f"{exp_name}.pth")
    torch.save(model.state_dict(), str(save_path))

    # Modify the last layer of the network
    # model.fc3 = nn.Linear(model.fc2.out_features, 10)

    # Training
    # TODO insert training code (you can use the train and test functions from deeplib)

    # Saving model
    # pretrained_folder = pathlib.Path('pretrained')
    # pretrained_folder.mkdir(exist_ok=True)
    # save_path = pretrained_folder.joinpath('state_name.pth')
    # torch.save(model.state_dict(), str(save_path))
