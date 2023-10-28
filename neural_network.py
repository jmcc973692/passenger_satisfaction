import json
import os
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.data_handling import prepare_data
from src.feature_engineering import perform_feature_engineering_nn
from src.hyperparameter_tuning import get_optimizer, tune_nn_parameters
from src.submission import load_sample_submission, save_submission, update_submission_structure
from src.TabularNN import TabularNN


def get_hyperparameters(train_x, train_y, x_val, y_val, x_test, y_test, device):
    file_path = "./hyperparameters/nn.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    best_params = tune_nn_parameters(train_x, train_y, x_val, y_val, x_test, y_test, device)

    with open(file_path, "w") as f:
        json.dump(best_params, f)

    return best_params


def compute_accuracy(predictions, labels):
    # Convert sigmoid outputs to binary predictions
    predicted_labels = (predictions > 0.5).float()
    correct = (predicted_labels == labels).float().sum()
    accuracy = correct / len(labels)
    return accuracy.item()


def main(train_path, test_path, sample_submission_path, submission_dir, device):
    # Load Data and preprocess 0's/ NaN values
    train_df, test_df = prepare_data(train_path, test_path)

    train_df = perform_feature_engineering_nn(train_df)
    test_df = perform_feature_engineering_nn(test_df)

    train_x = train_df.drop(columns=["Satisfaction_Rating", "id"])
    train_y = train_df["Satisfaction_Rating"]
    test_df = test_df.drop(columns=["id"])

    # Ensure both dataframes have the same column order
    train_x = train_x[sorted(train_x.columns)]
    test_df = test_df[sorted(test_df.columns)]

    # Convert data to PyTorch tensors
    train_x_tensor = torch.FloatTensor(train_x.values)
    train_y_tensor = torch.FloatTensor(train_y.values).unsqueeze(1)  # Convert to 2D tensor
    # Convert test data to tensor
    test_x_tensor = torch.FloatTensor(test_df.values)

    # First, split into (training + validation) and test sets
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        train_x_tensor, train_y_tensor, test_size=0.15, random_state=42
    )

    # Now, split the training + validation set into training and validation sets
    # 0.1765 of 0.85 is roughly 0.1
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.1765, random_state=42)

    best_params = get_hyperparameters(x_train, y_train, x_val, y_val, x_test, y_test, device)
    # Unpack the Best Parameters
    batch_size = best_params["batch_size"]
    dropout_rate = best_params["dropout"]
    factor = best_params["factor"]
    scheduler_patience = best_params["patience"]
    activation_functions = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "swish": nn.SiLU(),  # SiLU (Swish) was added in PyTorch 1.7.0 as nn.SiLU()
    }
    activation_func = activation_functions[best_params["activation"]]
    use_batch_norm = best_params["use_batch_norm"]

    # Create DataLoader Objects
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True, num_workers=10, persistent_workers=True
    )

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, pin_memory=True, num_workers=10, persistent_workers=True
    )

    input_dim = train_x.shape[1]
    model = TabularNN(
        input_dim, dropout_rate=dropout_rate, activation_func=activation_func, use_batch_norm=use_batch_norm
    )

    # Move model to the available CUDA device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model.parameters(), best_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=scheduler_patience, verbose=True
    )

    # Early stopping parameters
    patience = 20  # number of epochs to wait for improvement before terminating
    best_val_loss = float("inf")
    epochs_no_improve = 0
    epochs = 1000

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}], Training Loss: {running_loss/len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}], Validation Loss: {val_loss/len(val_loader):.4f}")

        # Step the Scheduler
        scheduler.step(val_loss)

        # Check if the validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            epochs_no_improve = 0  # Reset the Counter
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early Stopping!")
            break

    model.load_state_dict(best_model_weights)
    with torch.no_grad():
        model.eval()
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        test_predictions = model(x_test)
        test_accuracy = compute_accuracy(test_predictions, y_test)
        print(f"Final Test Set Accuracy: {test_accuracy:.4f}")

    # Save model
    timestamp = datetime.now().strftime("%m_%d_%Y_%H-%M")
    model_name = f"nn_model{timestamp}.pt"
    torch.save(model, os.path.join("./output/models", model_name))

    # Load sample submission
    sample_submission_df = load_sample_submission(sample_submission_path)

    # Make predictions on the test set
    model.eval()
    with torch.no_grad():
        test_x_tensor = test_x_tensor.to(device)
        test_outputs = model(test_x_tensor)
        # Convert probabilities to binary labels; using 0.5 as threshold

        y_pred_test = (test_outputs > 0.5).cpu().float().numpy().flatten()

    # Create Submission
    submission_file_name = f"nn_submission_{timestamp}.csv"
    submission_path = os.path.join(submission_dir, submission_file_name)
    # Update submission structure
    update_submission_structure(sample_submission_df, y_pred_test)
    save_submission(sample_submission_df, submission_path)

    print(f"Submission saved to {submission_path}")

    del x_train, x_val, y_train, y_val, x_test, y_test, train_x_tensor, train_y_tensor, x_trainval, y_trainval
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    sample_submission_path = "./input/sample_submission.csv"
    submission_dir = "./submission"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    main(train_path, test_path, sample_submission_path, submission_dir, device)
