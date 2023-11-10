import json
import os
import warnings
from datetime import datetime

# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.data_handling import prepare_data
from src.error_analysis import error_analysis
from src.feature_engineering import perform_feature_engineering_lgbm
from src.feature_selection import keep_best_features_only
from src.hyperparameter_tuning import (
    tune_lgbm_parameters,
    tune_rf_parameters,
    tune_xgb_parameters,
)
from src.submission import (
    load_sample_submission,
    save_submission,
    update_submission_structure,
)


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
    train_df = perform_feature_engineering_lgbm(train_df)
    test_df = perform_feature_engineering_lgbm(test_df)

    # Uncomment to output a new full feature set csv file including all of the feature engineering
    # train_df.to_csv("./output/train_full_feature_set.csv", index=False)
    # test_df.to_csv("./output/test_full_feature_set.csv", index=False)

    # train_df = keep_best_features_only(train_df)
    # test_df = keep_best_features_only(test_df)

    # train_df = drop_low_importance_features(train_df)
    # test_df = drop_low_importance_features(test_df)

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

    # Train the model
    model.fit(train_x, train_y)

    # Evaluate Average Cross-Validated accuracy
    scores = cross_val_score(
        model, train_x, train_y, cv=5, n_jobs=-1, scoring="accuracy"
    )
    print(f"Cross-Validation Accuracy Scores: {scores}")
    # Calculate and print the average accuracy over all folds
    avg_accuracy = np.mean(scores)
    print(f"Average cross-validated accuracy: {avg_accuracy:.4f}")

    # Evaluate model accuracy on training data
    accuracy = (model.predict(train_x) == train_y).sum() / len(train_y)
    print(f"Accuracy on training data: {accuracy:.4f}")

    # Output model importances
    importance_df = pd.DataFrame(
        {"Feature": train_x.columns, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    importance_df.to_csv("./output/feature-importance.txt", index=False, sep="\t")

    # Model Error Analysis
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4322)
    for train_idx, val_idx in skf.split(train_x, train_y):
        _, X_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
        _, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]

    wrong_predictions = error_analysis(X_val, y_val, model)
    wrong_predictions.to_csv("./output/error_analysis.csv", index=False)

    # Create a Submission with the New Model
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

    main(train_path, test_path, sample_submission_path, submission_dir, algorithm="xgb")
