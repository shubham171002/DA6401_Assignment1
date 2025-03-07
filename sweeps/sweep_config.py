import wandb
wandb.login()

sweep_config = {
    'method': 'random',  # Use random search to sample from the space.
    'program': 'sweeps/sweep_train.py',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {'values': [5, 10]},
        'num_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        'batch_size': {'values': [16, 32, 64]},
        'weight_init': {'values': ["random", "Xavier"]},
        'activation': {'values': ["sigmoid", "tanh", "ReLU"]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="DA6401_A1", entity="da24m020-iit-madras")
print("Sweep ID:", sweep_id)

