# Backprop

## `operation.py`
- each operation, given some in-flowing gradient, computes the gradients of its inputs by multiplying the in-flowing gradient by the operation's Jacobian

## `scratch_grad.py`

### `Variable._build_gradient(self)`
1. recursively computes gradients of all of its children in the computational graph
2. sums up the in-flowing gradient contributions by differentiating its consumer operations

### `Variable.backprop(self, gradient)`
1. Using the auxiliary method `Variable._build_gradient_all_parents(self, computed_vars_set)` it depth-first searches (DFS) the computational graph with reversed edges
2. On each discovered Variable (all ancestors) it recursively calls `._build_gradient(self)`
3. As a result, it computes the gradient of all reachable Variables from the `Variable` on which `.backprop` was called


## `test_scratch_grad.py`
- backprop unit-tests

```
$ python test_scratch_grad.py
...
----------------------------------------------------------------------
Ran 4 tests in 0.089s

OK 
```

## `train.py`
- generates random training set of size `train_set_size` with random labels
- creates fully connected classifier with 2 hidden layers of size `hidden_size` and `output_size` number of classification categories
- runs `epochs` numbers of gradient descent updates with learning rate `learning_rate`

### Output
#### **Hyperparameters**
```
    train_set_size = 16
    inp_size = 10
    hidden_size = 10
    output_size = 5
    epochs = 20
    learning_rate = 0.01
```


```
$ python train.py
Epoch: 1
ScratchNet output:  [0.00000 0.00000 0.00000 0.00636 0.99364]
ground_truth:  0

....

Epoch: 20
ScratchNet output:  [0.95176 0.00001 0.03901 0.00000 0.00922]
ground_truth:  0
ScratchNet output:  [0.42849 0.05696 0.45734 0.01638 0.04084]
ground_truth:  2
ScratchNet output:  [0.99250 0.00020 0.00720 0.00000 0.00010]
ground_truth:  0
ScratchNet output:  [0.95142 0.00001 0.04806 0.00000 0.00051]
ground_truth:  0
ScratchNet output:  [0.62664 0.00179 0.37043 0.00000 0.00114]
ground_truth:  0
ScratchNet output:  [0.00003 0.00000 0.00033 0.00000 0.99965]
ground_truth:  4
ScratchNet output:  [0.00035 0.00894 0.95047 0.02192 0.01832]
ground_truth:  2
ScratchNet output:  [0.00000 0.99754 0.00000 0.00000 0.00246]
ground_truth:  1
ScratchNet output:  [0.02154 0.00915 0.07780 0.00004 0.89146]
ground_truth:  4
ScratchNet output:  [0.00641 0.95742 0.00578 0.02900 0.00139]
ground_truth:  1
ScratchNet output:  [0.00050 0.00312 0.00634 0.00295 0.98708]
ground_truth:  4
ScratchNet output:  [0.45989 0.00644 0.49974 0.00000 0.03393]
ground_truth:  2
ScratchNet output:  [0.00157 0.22109 0.00771 0.76685 0.00278]
ground_truth:  3
ScratchNet output:  [0.20000 0.20000 0.20000 0.20000 0.20000]
ground_truth:  4
ScratchNet output:  [0.20000 0.20000 0.20000 0.20000 0.20000]
ground_truth:  2
ScratchNet output:  [0.00000 1.00000 0.00000 0.00000 0.00000]
ground_truth:  1
train loss=5.759520500696369
accuracy=87.5%
```