import os
from datetime import datetime

import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.data_handling import prepare_data
from src.feature_engineering import perform_feature_engineering_nn
from src.submission import load_sample_submission, save_submission, update_submission_structure
from src.TabularNN import TabularNN


def compute_accuracy(predictions, labels):
    # Convert sigmoid outputs to binary predictions
    predicted_labels = (predictions > 0.5).float()
    correct = (predicted_labels == labels).float().sum()
    accuracy = correct / len(labels)
    return accuracy.item()


def main(train_path, test_path, sample_submission_path, submission_dir):
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
    test_x_tensor = torch.FloatTensor(test_df.values)  # Assuming 'id' column is present in test_df

    # Split data
    x_train, x_val, y_train, y_val = train_test_split(train_x_tensor, train_y_tensor, test_size=0.2, random_state=42)

    # Create DataLoader Objects
    batch_size = 64  # You can adjust this value

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_dim = train_x.shape[1]
    model = TabularNN(input_dim)

    # Hyperparameters
    learning_rate = 0.001
    weight_decay = 1e-5

    # Early stopping parameters
    patience = 10  # number of epochs to wait for improvement before terminating
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Loss and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)

    epochs = 1000
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {running_loss/len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss/len(val_loader):.4f}")

        # Step the Scheduler
        scheduler.step(val_loss)

        # Check if the validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset the Counter
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early Stopping!")
            break

    with torch.no_grad():
        model.eval()

        train_predictions = model(train_x_tensor)
        train_accuracy = compute_accuracy(train_predictions, train_y_tensor)
        print(f"Final Training Accuracy: {train_accuracy:.4f}")

    # Save model
    torch.save(model.state_dict(), "./models/nn_model.pt")

    # Load sample submission
    sample_submission_df = load_sample_submission(sample_submission_path)

    # Make predictions on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_x_tensor)
        # Convert probabilities to binary labels; using 0.5 as threshold
        y_pred_test = (test_outputs > 0.5).float().numpy().flatten()

    # Create Submission
    timestamp = datetime.now().strftime("%m_%d_%Y_%H-%M")
    submission_file_name = f"nn_submission_{timestamp}.csv"
    submission_path = os.path.join(submission_dir, submission_file_name)
    # Update submission structure
    update_submission_structure(sample_submission_df, y_pred_test)
    save_submission(sample_submission_df, submission_path)

    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    sample_submission_path = "./input/sample_submission.csv"
    submission_dir = "./submission"

    main(train_path, test_path, sample_submission_path, submission_dir)
