import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data_handling import prepare_data
from src.feature_engineering import perform_feature_engineering_nn
from src.TabularNN import TabularNN


def main(train_path, test_path, sample_submission_path, submission_dir):
    # Load Data and preprocess 0's/ NaN values
    train_df, test_df = prepare_data(train_path, test_path)

    train_df = perform_feature_engineering_nn(train_df)
    test_df = perform_feature_engineering_nn(test_df)

    train_x = train_df.drop(columns=["Satisfaction_Rating", "id"])
    train_y = train_df["Satisfaction_Rating"]

    print(train_x.isnull().sum())
    # Convert data to PyTorch tensors
    train_x_tensor = torch.FloatTensor(train_x.values)
    train_y_tensor = torch.FloatTensor(train_y.values).unsqueeze(1)  # Convert to 2D tensor

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
    epochs = 20

    # Loss and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {1 - running_loss/len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {1 - val_loss/len(val_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "./models/nn_model.pt")


if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    sample_submission_path = "./input/sample_submission.csv"
    submission_dir = "./submission"

    main(train_path, test_path, sample_submission_path, submission_dir)
