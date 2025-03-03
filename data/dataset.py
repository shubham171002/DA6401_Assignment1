# Pre processing of fashin-MNIST dataset 
# Normalize the images

import numpy as np
from keras.datasets import fashion_mnist

def data_normalization():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    return X_train, y_train, X_test, y_test
