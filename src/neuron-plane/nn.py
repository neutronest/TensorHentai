#-*- coding:utf-8 -*-
import math
import random

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# return area [-1, 1]
def random_clamped():
    return random.random()*2-1


class Neuron():
    # neuron is the basic element of layer
    # a layer is constucted with multiple neurons
    
    def __init__(self):
        self.weights = []
        self.bias = 0.0
        self.res = 0.0

    def init_weights(self, n_neuron):
        for i in xrange(0, n_neuron):
            self.weights.append(random_clamped())

    def __repr__(self):
        return 'Neuron weight size:{}, bias value:{}'.format(self.weights, self.bias)

class Layer():
    def __init__(self, idx):
        self.idx = idx
        self.neurons = []

    def init_neurons(self, num_neuron, num_input):
        for i in xrange(0, num_neuron):
            neuron = Neuron()
            neuron.init_weights(num_input)
            self.neurons.append(neuron)

    def __repr__(self):
        return 'Layer ID:{}, Layer neuron size:{}'.format(self.idx, len(self.neurons))


class NeuronNetwork():
    def __init__(self):
        self.layers = []

    def init_neuron_network(self, n_input, n_hiddens, n_output):
        """
        Params:
        n_input: the number of input
          type: int
          example: 4
        n_hiddens: the list of num-of-hidden layers
          type: list of int
          example: [16, 32]
        n_output: the number of output
          type: int
          example: 4
        """
        idx = 0
        prev_neurons = 0
        layer = Layer(idx)
        layer.init_neurons(n_input, prev_neurons)
        prev_neurons = n_input
        self.layers.append(layer)
        idx += 1

        # add hidden layers
        for i in xrange(len(n_hiddens)):
            layer = Layer(idx)
            layer.init_neurons(n_hiddens[i], prev_neurons)
            prev_neurons = n_hiddens[i]
            self.layers.append(layer)
            idx += 1
        # add output layers
        layer = Layer(idx)
        layer.init_neurons(n_output, prev_neurons)
        prev_neurons = n_output
        self.layers.append(layer)
        idx += 1
        return

    def get_weights(self):
        data = {}
        data["weights"] = []
        data["network"] = []
        for layer in self.layers:
            data["network"].append(len(layer.neurons))
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    data["weights"].append(weight)
        return data

    def set_weights(self, data):
        self.layer = []
        weight_idx = 0
        idx = 0
        prev_neurons = 0
        for n_neurons in data.get("network"):
            layer = Layer(idx)
            layer.init_neurons(n_neurons, prev_neurons)
            for j in xrange(len(layer.neurons)):
                for k in xrange(len(layer.neurons[j].weights)):
                    layer.neurons[j].weights[k] = data["weights"]["weight_idx"]
                    weight_idx += 1
            prev_neurons = n_neurons
            idx += 1
        self.layers.append(layer)
        return

    def forward(self, input_data):
        for i in xrange(0, len(input_data)):
            self.layers[0].neurons[i].res = input_data[i]

        prev_neurons = self.layers[0]
        for i in xrange(1, len(self.layers)):
            for j in xrange(len(self.layers[i].neurons)):
                neuron = self.layers[i].neurons[j]
                sum_res = 0.0
                for k in xrange(len(neuron.weights)):
                    weight = neuron.weights[k]
                    #bias = neuron.bias[k]
                    res = sigmoid(weight * prev_neurons.neurons[k].res)
                    sum_res += res
                self.layers[i].neurons[j].res = sum_res
            prev_neurons = self.layers[i]

        out = []
        out_layer = self.layers[-1]
        for neuron in out_layer.neurons:
            out.append(neuron.res)
        return out

if __name__ == "__main__":
    print "hello world!"
    nn = NeuronNetwork()
    nn.init_neuron_network(4, [16], 3)
    w = nn.get_weights()
    out = nn.forward([1.21, 0.11, 3.22, 4.3])
    print out
    print w

