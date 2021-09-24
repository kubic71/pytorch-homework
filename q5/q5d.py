import numpy as np
import torch

from deeplib.datasets import load_mnist
from deeplib.training import train, test
from q5_helper import load_emnist, Network
from q5_finetune import finetune

# Pretrained model on Fashion-MNIST
# Finetune the last layer on EMNIST

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int, help='Finetuning batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Finetuning learning rate')
    parser.add_argument('--n_epoch', default=10, type=int, help='Number of finetuning epochs')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set, test_set = load_emnist('data/')

    # Init model
    model = Network().to(device)

    # Load model pretrained on Fashion-MNIST 
    model.load_state_dict(torch.load("pretrained/fashion_pretraining_n_epoch=20,bs=128,lr=0.001.pth"))

    exp_name = f"pretrain_Fashion-MNIST_finetune_last_layer_n_epoch={args.n_epoch},bs={args.batch_size},lr={args.lr}"
    finetune(model, train_set, test_set, exp_name, args.batch_size, args.lr, args.n_epoch, freeze=True)