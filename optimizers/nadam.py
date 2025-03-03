import numpy as np

def nadam_update(weights, biases, dW, dB, learning_rate, m, v, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
    for i in range(len(weights)):
        m[i] = beta1 * m[i] + (1 - beta1) * dW[i]
        v[i] = beta2 * v[i] + (1 - beta2) * (dW[i] ** 2)

        m_corrected = m[i] / (1 - beta1 ** t)
        v_corrected = v[i] / (1 - beta2 ** t)

        weights[i] -= learning_rate * (beta1 * m_corrected + (1 - beta1) * dW[i] / (1 - beta1)) / (np.sqrt(v_corrected) + epsilon)
        biases[i] -= learning_rate * dB[i]
