import numpy as np

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
