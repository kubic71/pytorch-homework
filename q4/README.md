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

