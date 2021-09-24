# Question 4
To overfit fully connected feed forward net on MNIST, run:
```
$ python q4.py
```

## Hyperparameters 
- `n_epochs=30`
- `batch_size = 2048`
    - We use large batch size, which is known to cause overfitting ([On Large-Batch Training for Deep Learning](https://arxiv.org/abs/1609.04836))
- `hidden_sizes = [4096]`
    - The feedforward net has only 1 large hidden layer
- `lr = 0.006`
    - constant learning rate for the Adam optimizer

## Training progress
![Loss and accuracy figure](q4_mnist_overfit_acc_loss_figure.png)


## Result
**After 30 epochs we achive**:
- `train_loss: 0.000203`
- `train_acc: 100%`
- `validation_loss: 0.071842`
- `validation_acc: 98.183334`

# Question 5

## Pretraining
The CNN is more expensive to train than our previous feedforward model. We therefore train it only for 20 epochs.
Loss and accuracy figures and metrics are saved in the "results" directory.

Pretrained model weights are saved in the "pretrained" directory.


### **MNIST**
```
$ python q5_pretrain.py --batch_size 128 --lr 0.001 --n_epoch 20 --dataset mnist
```

#### **Pretraining progress**
![CNN MNIST pretraining](results/mnist_pretraining_n_epoch=20,bs=128,lr=0.001.png)

### **Fashion-MNIST**
```
$ python q5_pretrain.py --batch_size 128 --lr 0.001 --n_epoch 20 --dataset fashion
```

![CNN Fashion-MNIST pretraining](results/fashion_pretraining_n_epoch=20,bs=128,lr=0.001.png)


### Pretraining results
|dataset|train accuracy|test accuracy|
|-|-|-|
|MNIST|99.752%|99.08|
|Fashion-MNIST|98.008%|91.04%|


## Transfer learning
We will finetune pretrained MNIST/Fashion-MNIST on a subset of EMNIST dataset.


|pretraining dataset|target dataset|finetuned layers|file|
|-|-|-|-|
|None|EMNIST (reduced)|All|q5a.py
|MNIST|EMNIST (reduced)|Last|q5b.py
|MNIST|EMNIST (reduced)|All|q5c.py
|Fashion-MNIST|EMNIST (reduced)|Last|q5d.py
|Fashion-MNIST|EMNIST (reduced)|All|q5e.py


### Hyperparameters
- `n_epoch=10`
- `batch_size=32`
- `lr=0.001`

### Figures

#### No pretraining, finetune all layers
**TODO**

#### MNIST pretraining, finetune last layer
![MNIST pretraining, finetune last layer](results/pretrain_MNIST_finetune_last_layer_n_epoch=10,bs=32,lr=0.001.png)

#### MNIST pretraining, finetune all layers
![MNIST pretraining, finetune all layers](results/pretrain_MNIST_finetune_all_layers_n_epoch=10,bs=32,lr=0.001.png)

#### Fashion-MNIST pretraining, finetune last layer
![Fashion-MNIST pretraining, finetune last layer](results/pretrain_Fashion-MNIST_finetune_last_layer_n_epoch=10,bs=32,lr=0.001.png)

#### Fashion-MNIST pretraining, finetune all layers
![Fashion-MNIST pretraining, finetune all layers](results/pretrain_Fashion-MNIST_finetune_all_layers_n_epoch=10,bs=32,lr=0.001.png)


### Results

|pretraining dataset|finetuned layers|train accuracy|test accuracy
|-|-|-|-|
|None|All|TODO|TODO|
|MNIST|Last|62.62%|58.90%|
|MNIST|All|97.42%|86.93%|
|Fashion-MNIST|Last|43.94%|40.57%|
|Fashion-MNIST|All|96.62%|83.25%|