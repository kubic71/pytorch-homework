import torch
from torch import nn
from deeplib.training import train, test
from q5_helper import get_trainable_params, freeze_model

import matplotlib.pyplot as plt

EMNIST_OUTPUT_CLASSES = 26

def finetune(model, emnist_train, emnist_test, exp_name, bs, lr, n_epoch, freeze=True):
    """Finetune given model on (reduced) EMNIST dataset
    args:
        model: CNN model pretrained on MNIST
        emnist_train: EMNIST train dataset
        emnist_test: EMNIST test dataset
        exp_name: filename prefix for the resulting .png figures and .txt metrics
        bs: batch size
        lr: learning rate
        n_epoch: number of finetuning epochs
        freeze: Whether to freeze all but the last layer
            - if True, only the last layer is trained
            - if False, all layers are finetuned
    """

    if freeze:
        # train only the last layer
        freeze_model(model)

    # TODO choice of initialization
    # PyTorch initializes linear layers from Uniform_Dist(-sqrt(k), sqrt(k)), where k=#in_features
    # But there are other initialization distributions (Xavier, Standard normal)
    # nice interactive post: https://www.deeplearning.ai/ai-notes/initialization/
    # but these custom initializations are probably needed only in special cases
    # Uniform init (default PyTorch way) is probably enough
    model.fc3 = nn.Linear(model.fc2.out_features, EMNIST_OUTPUT_CLASSES)

    optimizer = torch.optim.Adam(get_trainable_params(model), lr=lr)
    history = train(model, optimizer, emnist_train, n_epoch=n_epoch, batch_size=bs)
    train_acc = history.history["acc"][-1]

    test_acc = test(model, emnist_test, bs)

    print(f"EMNIST train accuracy: {train_acc}")
    print(f"EMNIST test accuracy: {test_acc}")

    history.display(show_fig=False)
    plt.tight_layout()
    plt.savefig(f"results/{exp_name}.png")

    with open(f"results/{exp_name}_metrics.txt", "w") as f:
        f.write(f"Train accuracy: {train_acc}\n")
        f.write(f"Test accuracy: {test_acc}\n")
