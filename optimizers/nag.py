def momentum_update(weights, biases, dW, dB, learning_rate, velocity, beta=0.9):
    for i in range(len(weights)):
        velocity[i] = beta * velocity[i] + (1 - beta) * dW[i]
        weights[i] -= learning_rate * velocity[i]
        biases[i] -= learning_rate * dB[i]
