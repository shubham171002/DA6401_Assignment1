import numpy as np
from utils.activations import relu, tanh, sigmoid, softmax
from utils.activations import relu_derivative, tanh_derivative, sigmoid_derivative
from utils.loss_functions import cross_entropy,mean_squared_error
from optimizers.sgd import sgd_update,momentum_update,nesterov_update,rmsprop_update,adam_update,nadam_update

class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, activation,weight_init='random'):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.output_size = output_size

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
        a = softmax(z)

        self.z.append(z)
        self.a.append(a)
        return a

    def backwardpass(self, X, y_true, learning_rate, optimizer, epoch):
        m = X.shape[0]
        y_pred = self.a[-1]  # Softmax output

        # Compute gradients for output layer
        dz = (y_pred - y_true) / m
        dw = np.dot(self.a[-2].T, dz)
        db = np.sum(dz, axis=0, keepdims=True)

        dW = [dw]
        dB = [db]
        dZ = [dz]

        # Backpropagation through hidden layers
        for i in range(self.hidden_layers - 1, 0, -1):
            dz = np.dot(dz, self.weights[i + 1].T) * self.activation_derivative(self.a[i - 1])
            dw = np.dot(self.a[i - 1].T, dz)
            db = np.sum(dz, axis=0, keepdims=True)

            dZ.insert(0, dz)
            dW.insert(0, dw)
            dB.insert(0, db)

        # First hidden layer
        dz = np.dot(dz, self.weights[1].T) * self.activation_derivative(self.a[0])
        dw = np.dot(X.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)

        dZ.insert(0, dz)
        dW.insert(0, dw)
        dB.insert(0, db)

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
        if loss_type == 'corss_entropy':
            return cross_entropy(y_true, y_pred)
        else:
            return mean_squared_error(y_true, y_pred)