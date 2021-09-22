import numpy as np


class Operation:
    def __init__(self, *inputs, name):
        self.inputs = list(inputs)  # List of Variable objects, of length 1 or 2 depending of the subclass
        self.name = name

        self.derivatives = {}  # Dictionary in which the key is a variable and the value is the gradient relative to this input.
        for input in self.inputs:
            input.add_consumer(self)
            self.derivatives[input] = None

    def apply(self, output_shape):
        from scratch_grad import \
            Variable  # Here to avoid circular import, see http://stackabuse.com/python-circular-imports/
        self.output = Variable(output_shape)
        self.output.operation = self
        return self.output

    def derivative_of_input(self, input, output_gradient):
        if input not in self.derivatives:
            raise ValueError("The operation '%s' has no input %s" % (str(self), str(input)))
        if self.derivatives[input] is None:
            self.derivate_inputs(output_gradient)
        return self.derivatives[input]

    def derivate_inputs(self, gradient):
        """
        Nothing to do here, implemented in subclasses
        Compute and stores the gradient relative to all inputs in the dictionary self.derivatives.
        """
        raise NotImplementedError()

    def __str__(self):
        return self.name


class AddOperation(Operation):
    def __init__(self, left, right):
        assert left.shape == right.shape
        super().__init__(left, right, name='+')

    def apply(self):
        output_value = self.inputs[0].value + self.inputs[1].value
        output_variable = super().apply(output_value.shape)  # Create the output Variable by calling the superclass.
        # IMPORTANT: the superclass does all the plumbing, always create the
        # output Variable this way.
        output_variable.value = output_value
        return output_variable

    def derivate_inputs(self, gradient):
        self.derivatives[self.inputs[0]] = gradient
        self.derivatives[self.inputs[1]] = gradient


class SubOperation(Operation):
    def __init__(self, left, right):
        assert left.shape == right.shape
        super().__init__(left, right, name='-')

    def apply(self):
        # TODO
        pass

    def derivate_inputs(self, gradient):
        # TODO
        pass


class MulOperation(Operation):
    def __init__(self, left, right):
        assert left.shape == right.shape
        super().__init__(left, right, name='*')

    def apply(self):
        # TODO
        pass

    def derivate_inputs(self, gradient):
        # TODO
        pass


class DivOperation(Operation):
    def __init__(self, num, denum):
        assert num.shape == denum.shape
        super().__init__(num, denum, name='/')

    def apply(self):
        # TODO
        pass

    def derivate_inputs(self, gradient):
        # TODO
        pass


class MatmulOperation(Operation):
    def __init__(self, left, right):
        left.value = self._to_2D_array(left.value)
        right.value = self._to_2D_array(right.value)
        super().__init__(left, right, name='matmul')

    def _to_2D_array(self, array):
        if len(array.shape) == 1:
            return np.expand_dims(array, axis=0)
        elif len(array.shape) == 2:
            return array
        else:
            raise ValueError('Unsupported array shape')

    def apply(self):
        # TODO
        pass

    def derivate_inputs(self, gradient):
        if len(gradient.shape) == 1:
            gradient = np.expand_dims(gradient, axis=0)  # SoftmaxLoss returns a 1D gradient

        # TODO
        pass


class ReluOperation(Operation):
    def __init__(self, input):
        super().__init__(input, name='relu')

    def apply(self):
        # TODO
        pass

    def derivate_inputs(self, gradient):
        # TODO
        pass


class SoftmaxLossOperation(Operation):
    def __init__(self, input, label):
        input.value = np.squeeze(input.value)
        super().__init__(input, name='softmax')
        self.label = label

    def apply(self):
        output_variable = super().apply((1,))

        input_value = self.inputs[0].value
        # For numerical stability, see equation 6.33 in Deep Learning, Goodfellow et al.
        input_exp = np.exp(input_value - np.max(input_value))
        exp_sum = np.sum(input_exp)

        self.softmax_output = input_exp / exp_sum
        output_variable.value = -np.log(self.softmax_output[self.label])
        return output_variable

    def derivate_inputs(self, _):
        # TODO
        pass
