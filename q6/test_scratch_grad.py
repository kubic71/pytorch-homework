import unittest

import numpy as np
import numpy.testing as npt
import torch
import torch.autograd as autograd
import torch.nn.functional as F

import scratch_grad as sg


def simple_equation(a, b):
    return (((a + b) + (a * b)) / a - a) @ b @ a  # @ is matrix multiplication


class TestScratchGrad(unittest.TestCase):
    def test_simple_equation_forward(self):
        a = autograd.Variable(torch.Tensor(2, 2).fill_(3.0), requires_grad=True)
        b = autograd.Variable(torch.eye(2) * 10.0, requires_grad=True)
        c = simple_equation(a, b)

        sg_a = sg.Variable((2, 2))
        sg_a.value = np.ones((2, 2)) * 3
        sg_b = sg.Variable((2, 2))
        sg_b.value = np.eye(2) * 10
        sg_c = simple_equation(sg_a, sg_b)

        npt.assert_almost_equal(sg_c.value, c.data.numpy())

    def test_simple_equation_back(self):
        a = autograd.Variable(torch.Tensor(2, 2).fill_(3.0), requires_grad=True)
        b = autograd.Variable(torch.eye(2) * 10.0, requires_grad=True)
        c = simple_equation(a, b)
        c.backward(torch.ones_like(c))

        sg_a = sg.Variable((2, 2))
        sg_a.value = np.ones((2, 2)) * 3
        sg_b = sg.Variable((2, 2))
        sg_b.value = np.eye(2) * 10
        sg_c = simple_equation(sg_a, sg_b)
        sg_c.backprop(np.ones_like(sg_c.value))

        npt.assert_almost_equal(sg_a.gradient, a.grad.data.numpy(), decimal=5)
        npt.assert_almost_equal(sg_b.gradient, b.grad.data.numpy(), decimal=5)

    def test_neural_net_without_loss(self):
        net_in = autograd.Variable(torch.Tensor(7).normal_(std=0.1))
        ground_truth_label = 1

        w1 = autograd.Variable(torch.Tensor(7, 7).normal_(std=0.1), requires_grad=True)
        w2 = autograd.Variable(torch.Tensor(7, 4).normal_(std=0.1), requires_grad=True)
        w3 = autograd.Variable(torch.Tensor(4, 3).normal_(std=0.1), requires_grad=True)
        layer1_out = F.relu(net_in @ w1)
        layer2_out = F.relu(layer1_out @ w2)
        layer3_out = layer2_out @ w3
        softmax_out = F.softmax(layer3_out)
        loss = - torch.log(softmax_out[ground_truth_label])  # NOTE: en pratique, on utilise F.cross_entropy
        # qui utilise log_softmax et nll_loss.

        sg_net_in = sg.Variable((7,))
        sg_net_in.value = net_in.data.numpy()
        sg_w1 = sg.Variable((7, 7))
        sg_w1.value = w1.data.numpy()
        sg_w2 = sg.Variable((7, 4))
        sg_w2.value = w2.data.numpy()
        sg_w3 = sg.Variable((4, 3))
        sg_w3.value = w3.data.numpy()

        sg_layer1_out = sg.relu(sg_net_in @ sg_w1)
        sg_layer2_out = sg.relu(sg_layer1_out @ sg_w2)
        sg_layer3_out = sg_layer2_out @ sg_w3

        layer3_out.backward(torch.ones_like(layer3_out))
        sg_layer3_out.backprop(np.ones_like(sg_layer3_out.value))

        npt.assert_almost_equal(np.squeeze(sg_layer2_out.value), layer2_out.data.numpy(), decimal=5)
        npt.assert_almost_equal(sg_w1.gradient, w1.grad.data.numpy(), decimal=5)
        npt.assert_almost_equal(sg_w2.gradient, w2.grad.data.numpy(), decimal=5)
        npt.assert_almost_equal(sg_w3.gradient, w3.grad.data.numpy(), decimal=5)

    def test_neural_net_with_loss(self):
        net_in = autograd.Variable(torch.Tensor(7).normal_(std=0.1))
        ground_truth_label = 1

        w1 = autograd.Variable(torch.Tensor(7, 7).normal_(std=0.1), requires_grad=True)
        w2 = autograd.Variable(torch.Tensor(7, 4).normal_(std=0.1), requires_grad=True)
        w3 = autograd.Variable(torch.Tensor(4, 3).normal_(std=0.1), requires_grad=True)
        layer1_out = F.relu(net_in @ w1)
        layer2_out = F.relu(layer1_out @ w2)
        layer3_out = layer2_out @ w3
        softmax_out = F.softmax(layer3_out)
        loss = - torch.log(softmax_out[ground_truth_label])  # NOTE: en pratique, on utilise F.cross_entropy
        # qui utilise log_softmax et nll_loss.
        # Cela a une plus grande stabilité numérique
        # (d'où le Warning à l'exécution)

        sg_net_in = sg.Variable((7,))
        sg_net_in.value = net_in.data.numpy()
        sg_w1 = sg.Variable((7, 7))
        sg_w1.value = w1.data.numpy()
        sg_w2 = sg.Variable((7, 4))
        sg_w2.value = w2.data.numpy()
        sg_w3 = sg.Variable((4, 3))
        sg_w3.value = w3.data.numpy()

        sg_layer1_out = sg.relu(sg_net_in @ sg_w1)
        sg_layer2_out = sg.relu(sg_layer1_out @ sg_w2)
        sg_layer3_out = sg_layer2_out @ sg_w3
        sg_loss = sg.softmax_loss(sg_layer3_out, ground_truth_label)

        loss.backward()
        sg_loss.backprop()

        npt.assert_almost_equal(sg_loss.value, loss.data.numpy())
        npt.assert_almost_equal(sg_w1.gradient, w1.grad.data.numpy(), decimal=5)
        npt.assert_almost_equal(sg_w2.gradient, w2.grad.data.numpy(), decimal=5)
        npt.assert_almost_equal(sg_w3.gradient, w3.grad.data.numpy(), decimal=5)


if __name__ == '__main__':
    unittest.main()
