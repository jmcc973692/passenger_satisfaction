import json
import os
import warnings
from datetime import datetime

import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.data_handling import drop_low_importance_features, prepare_data
from src.feature_engineering import perform_feature_engineering
from src.hyperparameter_tuning import tune_lgbm_parameters, tune_rf_parameters, tune_xgb_parameters
from src.submission import load_sample_submission, save_submission, update_submission_structure

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


def get_hyperparameters(algorithm, train_x, train_y):
    file_path = f"./hyperparameters/{algorithm}.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    if algorithm == "xgb":
        best_params = tune_xgb_parameters(train_x, train_y)
    elif algorithm == "rf":
        best_params = tune_rf_parameters(train_x, train_y)
    elif algorithm == "lgbm":
        best_params = tune_lgbm_parameters(train_x, train_y)

    with open(file_path, "w") as f:
        json.dump(best_params, f)

    return best_params


def main(train_path, test_path, sample_submission_path, submission_dir, algorithm):
    train_df, test_df = prepare_data(train_path, test_path)

    # Feature Engineering Steps
    train_df = perform_feature_engineering(train_df)
    test_df = perform_feature_engineering(test_df)

    train_df = drop_low_importance_features(train_df)
    test_df = drop_low_importance_features(test_df)

    # Drop the ID and target before training
    train_x = train_df.drop(columns=["Satisfaction_Rating", "id"])
    train_y = train_df["Satisfaction_Rating"]

    # Hyperparameter Tuning
    best_params = get_hyperparameters(algorithm, train_x, train_y)
    if algorithm == "xgb":
        model = xgb.XGBClassifier(**best_params)
    elif algorithm == "rf":
        model = RandomForestClassifier(**best_params)
    elif algorithm == "lgbm":
        model = LGBMClassifier(**best_params)

    print(f"Best parameters for {algorithm}: {best_params}")
    # Train the model with XGBoost
    model.fit(train_x, train_y)

    scores = cross_val_score(model, train_x, train_y, cv=5)
    print("Cross-validated scores:", scores)
    print("Average score:", scores.mean())

    # Evaluate model accuracy on training data
    accuracy = (model.predict(train_x) == train_y).sum() / len(train_y)
    print(f"Accuracy on training data: {accuracy:.4f}")

    # Output model importances
    importance_df = pd.DataFrame({"Feature": train_x.columns, "Importance": model.feature_importances_}).sort_values(
        by="Importance", ascending=False
    )

    importance_df.to_csv("./output/feature-importance.txt", index=False, sep="\t")

    sample_submission_df = load_sample_submission(sample_submission_path)

    test_x = test_df.drop(columns=["id"])
    y_pred = model.predict(test_x)

    timestamp = datetime.now().strftime("%m_%d_%Y_%H-%M")
    submission_file_name = f"submission_{timestamp}.csv"
    submission_path = os.path.join(submission_dir, submission_file_name)

    update_submission_structure(sample_submission_df, y_pred)
    save_submission(sample_submission_df, submission_path)


if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    sample_submission_path = "./input/sample_submission.csv"
    submission_dir = "./submission"

    main(train_path, test_path, sample_submission_path, submission_dir, algorithm="lgbm")
