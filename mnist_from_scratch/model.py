import numpy as np


def init_params(layer_sizes):
    weights = []
    bias = []
    for i in range(1, len(layer_sizes)):
        bias.append(np.zeros(layer_sizes[i]))
        weights.append(
            np.random.randn(layer_sizes[i - 1], layer_sizes[i])
            / np.sqrt(layer_sizes[i - 1])
        )
    return weights, bias


class Net:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward_with_cache(self, input):
        input = input.flatten()
        cache = [input]

        if len(self.weights) != len(self.bias):
            raise Exception(
                "Please initialize the model with the same number of weights and bias (the number of layers)"
            )
        n_layers = len(self.weights)

        for i in range(0, n_layers):
            input = np.matmul(input, self.weights[i]) + self.bias[i]
            cache.append(input)

            if i != n_layers - 1:
                input = np.maximum(input, 0)
                cache.append(input)
        return input, cache

    def __call__(self, input):
        logits, _ = self.forward_with_cache(input)
        return logits
