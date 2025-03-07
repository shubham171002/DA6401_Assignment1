import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.activations import relu, tanh, sigmoid, softmax
from utils.activations import relu_derivative, tanh_derivative, sigmoid_derivative
from utils.loss_functions import cross_entropy,mean_squared_error

from optimizers.sgd import sgd_update
from optimizers.momentum import momentum_update
from optimizers.nag import nesterov_update
from optimizers.rmsprop import rmsprop_update
from optimizers.adam import adam_update 
from optimizers.nadam import nadam_update

class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, activation,weight_init,loss_type='cross_entropy'):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.loss_type = loss_type

        # Activation functions
        activations = {"sigmoid": sigmoid, "tanh": tanh, "ReLU": relu}
        derivatives = {"sigmoid": sigmoid_derivative, "tanh": tanh_derivative, "ReLU": relu_derivative}
        
        self.activation = activations[activation]
        self.activation_derivative = derivatives[activation]

        # Initialize weights, biases, and optimization parameters
        self.weights = []
        self.biases = []
        self.velocity = []  # For Momentum & Nesterov
        self.cache = []  # For RMSprop
        self.m = []  # First moment (Adam & Nadam)
        self.v = []  # Second moment (Adam & Nadam)

        # Define layer sizes
        layer_sizes = [input_size] + [hidden_size] * hidden_layers + [output_size]
        
        # Function to initialize weights
        def initialize_weights(in_size, out_size, method):
            if method == "random":
                return np.random.randn(in_size, out_size) * 0.01  # Small random values
            elif method == "Xavier":
                return np.random.randn(in_size, out_size) * np.sqrt(1.0 / in_size)  # Xavier Initialization
            else:
                raise ValueError("Incorrect initialization")

        # Initialize weights and biases
        self.weights = [initialize_weights(layer_sizes[i], layer_sizes[i + 1], weight_init) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]

        # Initialize optimization parameters
        self.velocity = [np.zeros_like(w) for w in self.weights]  # For Momentum & Nesterov
        self.cache = [np.zeros_like(w) for w in self.weights]  # For RMSprop
        self.m = [np.zeros_like(w) for w in self.weights]  # Adam First Moment
        self.v = [np.zeros_like(w) for w in self.weights]  # Adam Second Moment

    def forwardpass(self, X):

        self.a = []
        self.z = []

        # Input to first hidden layer
        z = np.dot(X, self.weights[0]) + self.biases[0]
        a = self.activation(z)
        self.z.append(z)
        self.a.append(a)

        # Hidden layers
        for i in range(1, self.hidden_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.activation(z)
            self.z.append(z)
            self.a.append(a)

        # Last layer to output
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        if self.loss_type == 'cross_entropy':
            a = softmax(z)
        else:
            a = z

        self.z.append(z)
        self.a.append(a)
        return a

    def backwardpass(self, X, y_true, learning_rate, optimizer, epoch):
        m = X.shape[0]
        # Convert labels to one-hot encoding if needed
        if y_true.ndim == 1:
            y_true = np.eye(self.output_size)[y_true]
    
        # Total number of layers (hidden + output)
        L = len(self.weights)  # output layer index is L-1; hidden layers: 0 to L-2
    
        # ----- Output Layer -----
        # self.a[-1] is the output activation (after softmax if cross_entropy)
        a_out = self.a[-1]
        if self.loss_type == 'cross_entropy':
            dZ = (a_out - y_true) / m
        else:
            dZ = 2 * (a_out - y_true) / m
    
        # Gradients for the output layer weights and biases
        dW = [None] * L
        dB = [None] * L
        # Last hidden layer activation is self.a[-2]
        dW[L - 1] = np.dot(self.a[-2].T, dZ)
        dB[L - 1] = np.sum(dZ, axis=0, keepdims=True)
    
        # ----- Backpropagation for Hidden Layers -----
        # Propagate the gradient from the output layer
        dA = np.dot(dZ, self.weights[L - 1].T)
    
        # Loop backwards over hidden layers (from last hidden layer to first)
        for l in range(L - 2, -1, -1):
            # Compute dZ for the current hidden layer using the activation derivative
            dZ = dA * self.activation_derivative(self.z[l])
            # For the first hidden layer, use X as input; otherwise, use previous layer's activation
            if l == 0:
                dW[l] = np.dot(X.T, dZ)
            else:
                dW[l] = np.dot(self.a[l - 1].T, dZ)
            dB[l] = np.sum(dZ, axis=0, keepdims=True)
            # If not at the first hidden layer, propagate the gradient to the previous layer
            if l > 0:
                dA = np.dot(dZ, self.weights[l].T)
    
        # Update weights using the chosen optimizer
        self.update_weights(dW, dB, learning_rate, optimizer, epoch)


    def update_weights(self, dW, dB, learning_rate, optimizer, epoch):
        if optimizer == "sgd":
            sgd_update(self.weights, self.biases, dW, dB, learning_rate)
        elif optimizer == "momentum":
            momentum_update(self.weights, self.biases, dW, dB, learning_rate, self.velocity)
        elif optimizer == "nesterov":
            nesterov_update(self.weights, self.biases, dW, dB, learning_rate, self.velocity)
        elif optimizer == "rmsprop":
            rmsprop_update(self.weights, self.biases, dW, dB, learning_rate, self.cache)
        elif optimizer == "adam":
            adam_update(self.weights, self.biases, dW, dB, learning_rate, self.m, self.v, epoch)
        elif optimizer == "nadam":
            nadam_update(self.weights, self.biases, dW, dB, learning_rate, self.m, self.v, epoch)

    def compute_loss(self, loss_type,y_true, y_pred):
        if loss_type == 'cross_entropy':
            return cross_entropy(y_true, y_pred)
        else:
            return mean_squared_error(y_true, y_pred)
