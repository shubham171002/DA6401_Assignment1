from sklearn.metrics import accuracy_score
import numpy as np

def train(network, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, use_wandb=False, wandb_module=None, loss_type="mse",optimizer_params=None):
    if optimizer_params is None:
        optimizer_params = {}

    num_samples = X_train.shape[0]
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(epochs):
        # Shuffle training data at each epoch
        permutation = np.random.permutation(num_samples)
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = int(np.ceil(num_samples / batch_size))

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]
            
            # Flatten images if needed (e.g., from [batch, height, width] to [batch, height*width])
            X_batch_flat = X_batch.reshape(X_batch.shape[0], -1)
            
            # Forward pass
            y_pred = network.forwardpass(X_batch_flat)
            
            # Compute loss
            loss = network.compute_loss(loss_type, y_batch, y_pred)
            epoch_loss += loss
            
            # Compute training accuracy for this batch
            predictions = np.argmax(y_pred, axis=1)
            batch_acc = np.mean(predictions == y_batch)
            epoch_acc += batch_acc
            
            # Backward pass and weight update using the chosen optimizer
            network.backwardpass(X_batch_flat, y_batch, learning_rate, optimizer, epoch,optimizer_params)
        
        # Average the loss and accuracy over all batches
        epoch_loss /= num_batches
        epoch_acc /= num_batches

        # Validation pass
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        y_val_pred = network.forwardpass(X_val_flat)
        val_loss = network.compute_loss(loss_type, y_val, y_val_pred)
        val_predictions = np.argmax(y_val_pred, axis=1)
        val_acc = np.mean(val_predictions == y_val)

        # Log metrics into history
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} -- loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        
        # Log metrics to wandb if enabled
        if use_wandb and wandb_module:
            wandb_module.log({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": epoch_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
    
    return history

def evaluate(network, X_test, y_test, loss_type="cross_entropy", use_wandb=False, wandb_module=None):
    # Flatten test data
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Forward pass on test data
    y_pred_probs = network.forwardpass(X_test_flat)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute test accuracy
    test_acc = accuracy_score(y_test, y_pred)

    # Compute loss on test set
    test_loss = network.compute_loss(loss_type, y_test, y_pred_probs)

    # Print classification report
    print("Test Set Evaluation:")
    print(f"Test Accuracy: {test_acc*100:.2f}")
    print(f"Test Loss: {test_loss:.4f}")

    return {'test_accuracy': test_acc,'test_loss': test_loss}
