import math
class Neuron:
    def __init__(self, weights, bias, func):
        self.weights = weights
        self.bias = bias
        self.func = func
    def run(self, input_data):
        result = self.bias
        for i in range(len(input_data)):
            result += input_data[i] * self.weights[i]
        if self.func == "relu":
            return max(0, result)
        elif self.func == "sigmoid":
            return 1 / (1 + math.exp(-result))
        elif self.func == "tanh":
            return math.tanh(result)
        elif self.func == "binary_step":
            return 1 if result >= 0 else 0
        else:
            return None
    def changeBias(self, new_bias):
        self.bias = new_bias
    def changeWeights(self, new_weights):
        self.weights = new_weights
    def changeFunc(self, new_func):
        self.func = new_func