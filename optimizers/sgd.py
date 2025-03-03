def sgd_update(weights, biases, dW, dB, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * dW[i]
        biases[i] -= learning_rate * dB[i]
