import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import wandb

wandb.init(
    project="DA6401_A1",
    entity="da24m020-iit-madras",
)
# Load the dataset
(train_images, train_labels), (_,_) = fashion_mnist.load_data()

# Define the class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot" ]

plt.figure(figsize=(10, 5))
wandb_images = []

for i in range(10):
    # Get the first image that matches the class label
    idx = np.where(train_labels == i)[0][0]
    sample_image = train_images[idx]
    wandb_images.append(wandb.Image(sample_image, caption=class_names[i]))

    plt.subplot(2, 5, i + 1)
    plt.imshow(train_images[idx], cmap='gray')
    plt.title(class_names[i])
    plt.axis("off")

wandb.log({"Sample Images": wandb_images})
wandb.finish()

plt.tight_layout()
plt.show()