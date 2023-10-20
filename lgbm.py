import json
import os
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from src.data_handling import prepare_data
from src.feature_engineering import perform_feature_engineering, perform_feature_engineering_nn
from src.feature_selection import keep_best_features_only
from src.hyperparameter_tuning import tune_lgbm_parameters
from src.submission import load_sample_submission, save_submission, update_submission_structure


def get_hyperparameters(train_x, train_y):
    file_path = "./hyperparameters/lgbm.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    best_params = tune_lgbm_parameters(train_x, train_y)

    with open(file_path, "w") as f:
        json.dump(best_params, f)

    return best_params


def train_lgbm(train_x, train_y):
    best_params = get_hyperparameters(train_x, train_y)
    model = LGBMClassifier(**best_params)
    model.fit(train_x, train_y)

    return model


def save_lgbm_model(model, timestamp):
    model_file_name = f"lgbm_model_{timestamp}.pkl"
    model_path = os.path.join("./models", model_file_name)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    sample_submission_path = "./input/sample_submission.csv"
    submission_dir = "./submission"

    train_df, test_df = prepare_data(train_path, test_path)

    # Feature Engineering Steps
    train_df = perform_feature_engineering_nn(train_df)
    test_df = perform_feature_engineering_nn(test_df)

    # train_df = keep_best_features_only(train_df)
    # test_df = keep_best_features_only(test_df)

    # Drop the ID and target before training
    train_x = train_df.drop(columns=["Satisfaction_Rating", "id"])
    train_y = train_df["Satisfaction_Rating"]

    model = train_lgbm(train_x=train_x, train_y=train_y)

    # Evaluate Average Cross-Validated accuracy
    scores = cross_val_score(model, train_x, train_y, cv=5, n_jobs=-1, scoring="accuracy")
    print(f"Cross-Validation Accuracy Scores: {scores}")
    # Calculate and print the average accuracy over all folds
    avg_accuracy = np.mean(scores)
    print(f"Average cross-validated accuracy: {avg_accuracy:.4f}")

    # Evaluate model accuracy on training data
    accuracy = (model.predict(train_x) == train_y).sum() / len(train_y)
    print(f"Accuracy on training data: {accuracy:.4f}")

    # Output model importances
    importance_df = pd.DataFrame({"Feature": train_x.columns, "Importance": model.feature_importances_}).sort_values(
        by="Importance", ascending=False
    )

    importance_df.to_csv("./output/feature-importance.txt", index=False, sep="\t")

    # Create a Submission with the New Model
    sample_submission_df = load_sample_submission(sample_submission_path)

    test_x = test_df.drop(columns=["id"])
    y_pred = model.predict(test_x)

    timestamp = datetime.now().strftime("%m_%d_%Y_%H-%M")
    submission_file_name = f"submission_{timestamp}.csv"
    submission_path = os.path.join(submission_dir, submission_file_name)

    update_submission_structure(sample_submission_df, y_pred)
    save_submission(sample_submission_df, submission_path)

    save_lgbm_model(model, timestamp=timestamp)
