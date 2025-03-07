import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from model.neural_network import FeedforwardNeuralNetwork
from model.model_training import train
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

best_hyperparams = {
    "num_layers": 4,
    "hidden_size": 128,
    "activation": "ReLU",
    "batch_size": 32,
    "epochs": 5, 
    "optimizer": "rmsprop",
    "learning_rate": 0.001,
    "weight_decay": 0.5,
    "weight_init": "Xavier"
}


# Load the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0

# Flatten images for the neural network
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# One-hot encode the labels
y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)


# Initialize the model with best hyperparameters
model = FeedforwardNeuralNetwork(
    input_size=28*28,
    hidden_layers=best_hyperparams["num_layers"],
    hidden_size=best_hyperparams["hidden_size"],
    output_size=10,
    activation=best_hyperparams["activation"],
    weight_init=best_hyperparams["weight_init"]
)

# Training the model on full training data
train(
    model,
    optimizer=best_hyperparams["optimizer"],
    X_train=X_train_flat,
    y_train=y_train,
    X_val=X_train_flat[:5000],  # Using part of training data for validation
    y_val=y_train[:5000],
    epochs=best_hyperparams["epochs"],
    batch_size=best_hyperparams["batch_size"],
    learning_rate=best_hyperparams["learning_rate"],
    use_wandb=True
)

import numpy as np

# Forward pass on test data
y_pred = model.forwardpass(X_test_flat)

# Convert probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Compute test accuracy
test_accuracy = np.mean(y_pred_labels == y_test)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")


# Fashion-MNIST class labels
class_labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)

# Convert confusion matrix to row-wise percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot the confusion matrix in percentage
fig = plt.figure(figsize=(10, 8),dpi=120)
plt.imshow(cm_percentage, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix (%) - Fashion MNIST", fontsize=16)
plt.colorbar()

# Create tick marks for 10 classes (0-9)
tick_marks = np.arange(10)
plt.xticks(tick_marks, tick_marks, rotation=45, fontsize=12)
plt.yticks(tick_marks, tick_marks, fontsize=12)

# Add text annotations to each cell with percentages
thresh = cm_percentage.max() / 2.
for i in range(cm_percentage.shape[0]):
    for j in range(cm_percentage.shape[1]):
        plt.text(j, i, f"{cm_percentage[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm_percentage[i, j] > thresh else "black",
                 fontsize=12)

plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.show()

# Log the test accuracy and confusion matrix to W&B
import wandb

wandb.init(project="DA6401_A1", entity="da24m020-iit-madras", name="best_model_evaluation")
wandb.log({"Confusion Matrix": wandb.Image(fig)})

wandb.finish()