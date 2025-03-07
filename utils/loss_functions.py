import numpy as np

def cross_entropy(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = np.eye(y_pred.shape[1])[y_true]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
