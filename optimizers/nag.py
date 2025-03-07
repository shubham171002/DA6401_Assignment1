def nesterov_update(weights, biases, dW, dB, learning_rate, velocity, beta=0.9):
    for i in range(len(weights)):
        v_prev = velocity[i].copy()
        velocity[i] = beta * velocity[i] + (1 - beta) * dW[i]
        weights[i] -= learning_rate * (beta * v_prev + (1 - beta) * dW[i])
        biases[i] -= learning_rate * dB[i]
        
