
import numpy as np
import random

np.random.seed(0)

# A class representing a neural network define by the size of each layer and the number of layers.
# Most of the calculations utilize matrices.
# Definitions:
# - n_outputs is the number of neurons on a layer(L); n_inputs is the number of neurons in the previous layer(L-1).
# - Each layer(L) has a weight matrix and a bias column vector to compute the outputs of layer(L) from layer(L-1)
# - Each layer is represented as a matrix of (n_outputs * 1).
# - For each layer(L) except the first, the weight matrix is (n_outputs * n_inputs);
#    W_(i, j) is the weight connecting neuron i on layer(L) and neuron j on layer(L-1).
# - For each layer(L) except the first, the bias is (n_outputs * 1) column vector.
class Network:
    def __init__(self, sizes):
        # list of number of neurons on each layer, from layer 1 to final layer
        # e.g. the number at index 0 represents the size of layer 1
        self.sizes = sizes
        # number of layers
        self.n_layers = len(sizes)
        # list of layer objects except first layer (the raw input layer)
        self.layers = [Layer(n_inputs, n_outputs) for n_inputs, n_outputs in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, learn_rate):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs. x and y are supposed to be vectors"""
        n = len(training_data)
        # epochs is the number of times to iterate the learning process using a mini-batch
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.learn_mini_batch(mini_batch, learn_rate)

            print("Epoch {0} complete".format(j))

    def learn_mini_batch(self, mini_batch, learn_rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)" """
        for x, y in mini_batch:
            self.backprop(x, y)
        for layer in self.layers:
            layer.weights = layer.weights - (learn_rate / len(mini_batch)) * layer.nabla_w
            layer.biases = layer.biases - (learn_rate / len(mini_batch)) * layer.nabla_b

    def backprop(self, x, y):
        """Backpropagation"""
        # feedforward
        # set the activation of the second layer and the rest
        self.layers[0].feedforward(x)
        for i in range(1, self.n_layers-1, 1):
            self.layers[i].feedforward(self.layers[i-1].outputs)

        # backward pass
        lastlayer = self.layers[-1]
        # delta is partial derivative of cost function with respect to the weighted sum
        delta = np.multiply(self.cost_derivative(lastlayer.outputs, y), sigmoid_prime(lastlayer.weighted_sum))
        # addition is used here because it is intended to be adding the partial derivatives from various
        # samples in the mini batch together
        lastlayer.nabla_b += delta
        lastlayer.nabla_w += np.dot(delta, self.layers[-2].outputs.T)

        # propagates from second last layer to second layer sequentially
        for layer_index in range(self.n_layers-3, -1, -1):
            layer = self.layers[layer_index]
            nextlayer = self.layers[layer_index+1]
            weighted_sum = layer.weighted_sum
            sp = sigmoid_prime(weighted_sum)
            # a matrix multiplication followed by an element-wise product
            delta = np.multiply(np.dot(nextlayer.weights.T, delta), sp)
            # addition is used here because it is intended to be adding the partial derivatives from various
            # samples in the mini batch together
            layer.nabla_b += delta
            if layer_index != 0:
                layer.nabla_w += np.dot(delta, self.layers[layer_index-1].outputs.T)
            else:
                layer.nabla_w += np.dot(delta, x.T)

    def cost_derivative(self, output_activations, expected_activations):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - expected_activations

    def result_after_trained(self, x):
        """this method is supposed to be used after the network was fully trained.
        pass an input vector x representing the raw data, and this method would output
        the index of the highest activation of the final layer in the neural network"""
        self.layers[0].feedforward(x)
        for i in range(1, self.n_layers-1, 1):
            self.layers[i].feedforward(self.layers[i - 1].outputs)
        return self.layers[-1].outputs

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

# this class defines a layer of neurons
class Layer:
    # n_outputs is the number of neurons on this layer
    # n_inputs is the number of neurons on the previous layer
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_outputs, n_inputs)  # dimensions: n_inputs * n_outputs
        self.biases = np.random.randn(n_outputs, 1)  # dimensions: n_outputs * 1

        # these will be updated during feeding forward
        self.weighted_sum = None
        # outputs are all activations of this layer
        self.outputs = None
        # partial derivatives of cost function with respect to weights and biases of this layer
        # these will be updated during backpropagation
        self.nabla_b = np.zeros(self.biases.shape)
        self.nabla_w = np.zeros(self.weights.shape)

    def feedforward(self, inputs):
        self.weighted_sum = np.dot(self.weights, inputs) + self.biases
        self.outputs = sigmoid(self.weighted_sum)

# Rough Test
sizes = [2, 2, 2, 1]
net = Network(sizes)
x1 = np.random.randn(1, 2).T
y1 = np.random.randn(1, 1).T
x2 = np.random.randn(1, 2).T
y2 = np.random.randn(1, 1).T
training_data = [[x1, y1], [x2, y2]]
net.SGD(training_data, 40, 1, 3)
print(net.result_after_trained(np.random.randn(1, 2).T))
