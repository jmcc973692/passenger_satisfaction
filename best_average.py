import os
from datetime import datetime

import joblib
import torch

from src.data_handling import prepare_data
from src.feature_engineering import perform_feature_engineering, perform_feature_engineering_nn
from src.feature_selection import keep_best_features_only
from src.submission import load_sample_submission, save_submission, update_submission_structure

if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    sample_submission_path = "./input/sample_submission.csv"
    submission_dir = "./submission"

    train_df, test_df = prepare_data(train_path=train_path, test_path=test_path)

    test_df_lgbm = perform_feature_engineering_nn(test_df.copy())
    # test_df_lgbm = keep_best_features_only(test_df_lgbm)
    test_x_lgbm = test_df_lgbm.drop(columns=["id"])

    test_df_nn = perform_feature_engineering_nn(test_df.copy())
    test_df_nn = test_df_nn.drop(columns=["id"])

    test_x_tensor = torch.FloatTensor(test_df_nn.values)

    # Load LGBM model
    with open("./models/lgbm_model.pkl", "rb") as f:
        lgbm_model = joblib.load(f)

    # Load Neural Network model (assuming it's saved as a PyTorch model)
    nn_model = torch.load("./models/nn_model.pt")
    nn_model.eval()  # Set the neural network to evaluation mode

    # Predictions for LGBM
    lgbm_predictions = lgbm_model.predict_proba(test_x_lgbm)[:, 1]

    # Predictions for Neural Network (assuming you're using PyTorch)
    with torch.no_grad():
        nn_output = nn_model(test_x_tensor)
        nn_predictions = torch.sigmoid(nn_output).numpy().squeeze()

    alpha = 0.6
    weighted_predictions = alpha * lgbm_predictions + (1 - alpha) * nn_predictions

    binary_avg_predictions = (weighted_predictions > 0.55).astype(int)

    sample_submission_df = load_sample_submission(sample_submission_path=sample_submission_path)
    update_submission_structure(sample_submission_df, binary_avg_predictions)

    timestamp = datetime.now().strftime("%m_%d_%Y_%H-%M")
    submission_filename = f"avg_submission_{timestamp}.csv"
    submission_filepath = os.path.join(submission_dir, submission_filename)

    save_submission(sample_submission_df, submission_filepath)
