import argparse
import wandb
import numpy as np
from keras.datasets import fashion_mnist, mnist
from model.neural_network import FeedforwardNeuralNetwork
from model.model_training import train,evaluate

def main():
    parser = argparse.ArgumentParser(
        description="Train a feedforward neural network and log experiments to Weights & Biases"
    )
    # Wandb settings
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_A1",
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="da24m020-iit-madras",
                        help="Wandb Entity used to track experiments in Weights & Biases dashboard")
    # Dataset
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist",
                        help='Dataset to use: "mnist" or "fashion_mnist"')
    # Training parameters
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy",
                        help="Loss function to use.")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="rmsprop",
                        help="Optimizer to use.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.5,
                        help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.5,
                        help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                        help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                        help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6,
                        help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0,
                        help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier",
                        help="Weight initialization method.")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4,
                        help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128,
                        help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="ReLU",
                        help="Activation function to use.")

    args = parser.parse_args()

    optimizer_params = {
    "momentum": args.momentum,
    "beta": args.beta,
    "beta1": args.beta1,
    "beta2": args.beta2,
    "epsilon": args.epsilon,
    "weight_decay": args.weight_decay,
}

    # Initialize wandb with the parsed configuration
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)  # Convert argparse Namespace to dictionary
    )
    config = wandb.config

    # Set a descriptive run name for clarity in the dashboard
    run.name = (
        f"hl_{config.num_layers}_bs_{config.batch_size}_ac_{config.activation}_"
        f"hs_{config.hidden_size}_opt_{config.optimizer}_lr_{config.learning_rate}_"
        f"wd_{config.weight_decay}_wi_{config.weight_init}"
    )

    # Load the chosen dataset
    if config.dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif config.dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the image data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reserve 10% of the training data as validation set
    split_idx = int(0.9 * X_train.shape[0])
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]

    # Flatten images for the feedforward neural network
    input_size = X_train.shape[1] * X_train.shape[2] 
    X_train = X_train.reshape(-1, input_size)
    X_val = X_val.reshape(-1, input_size)
    X_test = X_test.reshape(-1, input_size)

    # Set output size (number of classes)
    output_size = 10

    # Initialize the neural network with the specified configuration
    network = FeedforwardNeuralNetwork(
        input_size=input_size,
        hidden_layers=config.num_layers,
        hidden_size=config.hidden_size,
        output_size=output_size,
        activation=config.activation,
        weight_init=config.weight_init,
        loss_type=config.loss
    )

    # Train the network and log metrics to wandb
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
        optimizer_params=optimizer_params,
        use_wandb=True,
        wandb_module=wandb,
        loss_type=config.loss
    )

    evaluation = evaluate(
    network=network,
    X_test=X_test,
    y_test=y_test,
    loss_type="cross_entropy"
    )


if __name__ == "__main__":
    main()
