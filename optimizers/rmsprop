import numpy as np

def rmsprop_update(weights, biases, dW, dB, learning_rate, cache, beta=0.9, epsilon=1e-8):
    for i in range(len(weights)):
        cache[i] = beta * cache[i] + (1 - beta) * (dW[i] ** 2)
        weights[i] -= learning_rate * dW[i] / (np.sqrt(cache[i]) + epsilon)
        biases[i] -= learning_rate * dB[i]
