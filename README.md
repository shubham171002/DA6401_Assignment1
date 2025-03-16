# DA6401_Assignment1  
This is the Repositary for Assignment 1 of Deep Learning (DA6401) in which FeedForward Neural Network has been implemented on fashion_MNIST dataset. Below is the detailed instructions and explanation for this assignment
## Project Structure 
```
DA6401_Assignment1/
├── model/                 # Core implementation module
    ├── neural_network.py  # Implementation of neural network
    ├── model_training.py  # Training and evaluation of the built network
├── utils/
    ├── activation.py      # Implementation of activation functions
    ├── loss_functions.py  # Implementation of loss functions
├── optimizers/            # Implementation of optimizers
    ├── adam.py
    ├── momentum.py
    ├── nag.py
    ├── nadam.py
    ├── sgd.py
    ├── rmsprop.py
├── sweeps/                # Sweep training with wandb ( Hyperparameter tuning )     
    ├── swep_config.py
    ├── sweep_train.py                                 
├── train.py               # Main training script with command-line arguments
└── wandb/                 # WandB logs and visualization
```
## Dataset 
The Fashion-MNIST dataset consists of 70,000 grayscale images of 28x28 pixels, divided into 10 classes representing various fashion items (e.g., T-shirts, trousers, etc.). The dataset is split into training and testing set.

## Task Details
*  Implement a multi-layer Feed Forward Neural Network from scratch.
*  Experiment with various configurations including number of hidden layers, hidden layer sizes, activation functions, optimization algorithms, and hyperparameters tuning.
*  Conduct hyperparameter tuning using Weights & Biases (WandB) sweeps.
*  Track experiments and performance metrics using WandB.
*  Visualize results and analyze hyperparameters.
*  Provide comparative analysis of different optimizers and loss functions (Cross-Entropy vs. Squared Error).

## Used Libraries:
*  Numpy: For numerical computations and array manipulation.
*  Matplotlib & Seaborn: Visualization and plotting.
*  scikit-learn: Metrics (e.g., confusion matrix, accuracy).
*  Keras/TensorFlow: Data handling (Fashion-MNIST, MNIST dataset loading).
*  Weights & Biases (wandb): Experiment tracking and logging.

## How to Run ?
### Step 1 : Install the required libraries
Command: ```pip install -r requirements.txt```
### Step 2 : Run the main script 
Command: ```python train.py```
Note that there are several other arguments which can be passed while running the above command like wandb_entity and wand_project which are set defaut and no need to change that, set of hyperparameters which are also set default (Measured for the model giving the highest validation accuracy) but can be passed according to suitability.
Following is the set of main hyperparameters which can be passed as arguments:
*  --epochs : No of epochs
*  --loss : Loss function (cross entropy or MSE)
*  --learning_rate : Learning rate
*  --activation : Activation function
*  --num_layes : No of hidden layes
*  --hidden_size : No of neurons in each hidden layer
*  --weight_init : Weight initialization (random or xavier)
*  --weight_decay : Weight decay used by optimizer

## Experiment Tracking with Wandb
The experiments were tracked using Weights & Biases, and detailed visualizations and insights can be found at the following report:
https://wandb.ai/da24m020-iit-madras/DA6401_A1/reports/DA6401-Assignment-1-Report--VmlldzoxMTY3ODMwMw?accessToken=mxa6u89d2pelvl20xs6jmlyz46c13amo2y1dv8opuk1zr62gcovtuhdgonk5xngi

## Some Key Observations 
1) fashion_MNIST dataset
   *  Highest validation accuracy achieved : ~87%
   *  Highest test accuracy achieved : ~85%
2) MNIST dataset
   * Highest training accuracy (for the hyperparams choosed based on the obvservations on fashion_MNIST) : 99.16%
   * Highest test accuracy : 97.25%
3) Graphs for accuracy and loss are smooth smooth for cross entropy where as fluctuation has been observed with MSE loss
4) RmsProp optimizer perfomed best among all the optimizers consistenly.

### Some Additional Information 
*  ```Q1.py``` file contains the implementation for generating sample images for all the classes in fashion_MNIST dataset
*  ```Q7.py``` file contains the implementaiton for generating confusion matrix for evaluation on test set of fashion_MNIST
