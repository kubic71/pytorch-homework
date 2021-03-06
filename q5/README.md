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

### Running the experiments
Example of running the q5b.py with the mentioned hyperparameters.
```
$  python q5b.py --lr 0.001 --batch_size 32 --n_epoch 10
```


### Figures

#### No pretraining, finetune all layers
![No pretraining, finetune all layers](results/no_pretraining_finetune_all_layers_n_epoch=10,bs=32,lr=0.001.png)

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
|None|All|94.48%|85.29%|
|MNIST|Last|62.62%|58.90%|
|MNIST|All|97.42%|86.93%|
|Fashion-MNIST|Last|43.94%|40.57%|
|Fashion-MNIST|All|96.62%|83.25%|


## Discussion
### Choice of the distribution to initialize the last layer
PyTorch by default uses centered uniform [-k, k] distribution (where k is the square root of the input size) for linear layer weights initialization.
It may not be the optimal intialization distribution, but [this stackoverflow post](https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch) makes it seem like there isn't a big difference compared to the normal distribution initialization. Tuning the hyperparameters will probably have a larger impact on the performance.

### The impact of the number of unfrozen layers
Training only the last linear layer makes the model underfit badly, because:
- we are training only for 10 epochs
- the frozen part of the network isn't optimized for EMNIST and may not supply all the essential features for EMNIST classification
- the last linear layer has small capacity (compared to the whole CNN)

### The impact of the difference in appearance of the source domain
MNIST dataset contains digits, which are more semantically similar to alphabet letters than some pictures of clothes. So as expected, pretraining on MNIST yields better results compared to pretraining on Fashion-MNIST.

### Pretraining didn't really help
What is maybe a little bit surprising is the fact, that the best pretraining result obtained is approximately as good as not using pretraining at all. 
