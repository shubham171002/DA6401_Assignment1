from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import wandb
import numpy as np
from keras.datasets import fashion_mnist

# Import  FeedforwardNeuralNetwork class
from model.neural_network import FeedforwardNeuralNetwork  
# Import the revised train and evaluate functions
from model.model_training import train

def sweep_train():
    # Initialize wandb and load configuration
    run = wandb.init(
        project="DA6401_A1",
        entity="da24m020-iit-madras",
    )
    config = wandb.config

    # Set a descriptive run name based on key hyperparameters
    run.name = (
        f"hl_{config.num_layers}_bs_{config.batch_size}_ac_{config.activation}_"
        f"hs_{config.hidden_size}_opt_{config.optimizer}_lr_{config.learning_rate}_"
        f"wd_{config.weight_decay}_wi_{config.weight_init}"
    )
    # Load Fashion-MNIST data
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reserve 10% of training data as validation data
    split_idx = int(0.9 * X_train.shape[0])
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]

    # Set input and output sizes
    input_size = 28 * 28
    output_size = 10

    # Initialize network using hyperparameters from wandb.config
    network = FeedforwardNeuralNetwork(
        input_size=input_size,
        hidden_layers=config.num_layers,
        hidden_size=config.hidden_size,
        output_size=output_size,
        activation=config.activation,
        weight_init=config.weight_init
    )

    # Train the network, logging metrics to wandb
    history = train(
        network,
        optimizer=config.optimizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        use_wandb=True,
        wandb_module=wandb,
        loss_type="cross_entropy"
    )

if __name__ == '__main__':
    sweep_train()