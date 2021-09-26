from operation import *


class Variable:
    _var_counter = 0

    def __init__(self, shape, name=None):
        self._value = np.zeros(shape)
        self.name = name if name is not None else self._gen_var_name()
        self.gradient = None
        self.operation = None  # The operation that created this Variable (None for inputs and parameters)
        self.consumer_operations = []  # List of operations that use this Variable.
        # Updated automatically by the constructor of Operation

    @property
    def value(self):
        return self._value

    @property
    def shape(self):
        return self._value.shape

    @value.setter
    def value(self, x):
        self._value = x

    def add_consumer(self, c):
        self.consumer_operations.append(c)

    def _build_gradient(self):  # Inspired by algorithm 6.6 (p.210) of the Deep Learning book
        if self.gradient is not None:
            return

        self.gradient = np.zeros_like(self._value)

        # compute gradient for all children
        for c_op in self.consumer_operations:
            c_op.output._build_gradient()

            # compute the gradient of consumer operation with respect to its inputs
            c_op.derivate_inputs(c_op.output.gradient)

            # add the gradient flowing from the consumer
            self.gradient += c_op.derivatives[self]


    def _build_gradient_all_parents(self, computed_vars_set):
        if self.operation is not None:
            parents = self.operation.inputs
            for p in parents:
                if p not in computed_vars_set:
                    computed_vars_set.add(p)
                    p._build_gradient()
                    p._build_gradient_all_parents(computed_vars_set)

    def backprop(self, gradient=None):
        if gradient is not None:
            assert gradient.shape == self._value.shape
            self.gradient = gradient
        computed_vars_set = {self}
        self._build_gradient_all_parents(computed_vars_set)

    def _gen_var_name(self):
        name = 'var_%d' % Variable._var_counter
        Variable._var_counter += 1
        return name

    def __str__(self):
        return self.name

    def __add__(self, other):

        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __matmul__(self, other):
        return matmul(self, other)


def add(left, right):
    return AddOperation(left, right).apply()


def sub(left, right):
    return SubOperation(left, right).apply()


def mul(left, right):
    return MulOperation(left, right).apply()


def div(num, denum):
    return DivOperation(num, denum).apply()


def matmul(left, right):
    return MatmulOperation(left, right).apply()


def relu(x):
    return ReluOperation(x).apply()


def softmax_loss(x, label):
    return SoftmaxLossOperation(x, label).apply()
