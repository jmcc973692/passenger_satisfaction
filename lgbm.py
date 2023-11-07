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
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data_handling import prepare_data
from src.error_analysis import error_analysis
from src.feature_engineering import perform_feature_engineering_lgbm
from src.feature_selection import keep_best_features_only
from src.hyperparameter_tuning import tune_lgbm_parameters
from src.optimal_threshold import find_optimal_threshold, threshold_scorer
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
    model = LGBMClassifier(**best_params, verbose=-1)
    model.fit(train_x, train_y)

    return model


# # Function to find the optimal threshold based on the Precision-Recall curve
# def find_optimal_threshold(y_true, y_probs):
#     precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

#     # Remove the last precision and recall values (which correspond to threshold of 1)
#     precision, recall = precision[:-1], recall[:-1]

#     # Calculate the geometric mean of precision and recall for each threshold
#     geometric_mean = np.sqrt(precision * recall)

#     # Find the index of the highest geometric mean
#     optimal_idx = np.argmax(geometric_mean)

#     # Find the optimal threshold corresponding to the highest geometric mean
#     optimal_threshold = thresholds[optimal_idx]

#     return optimal_threshold, precision[optimal_idx], recall[optimal_idx]


def save_lgbm_model(model, timestamp):
    model_file_name = f"lgbm_model_{timestamp}.pkl"
    model_path = os.path.join("./output/models", model_file_name)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    sample_submission_path = "./input/sample_submission.csv"
    submission_dir = "./submission"

    train_df, test_df = prepare_data(train_path, test_path)

    # Feature Engineering Steps
    train_df = perform_feature_engineering_lgbm(train_df)
    test_df = perform_feature_engineering_lgbm(test_df)

    # train_df = keep_best_features_only(train_df)
    # test_df = keep_best_features_only(test_df)

    # Drop the ID and target before training
    train_x = train_df.drop(columns=["Satisfaction_Rating", "id"])
    train_y = train_df["Satisfaction_Rating"]

    # Cross-Validation before applying optimal threshold
    model = train_lgbm(train_x=train_x, train_y=train_y)
    scores_before_threshold = cross_val_score(model, train_x, train_y, cv=10, n_jobs=-1, scoring="accuracy")
    print(f"Cross-Validation Accuracy Scores Before Threshold: {scores_before_threshold}")
    print(f"Average cross-validated accuracy before threshold: {np.mean(scores_before_threshold):.4f}")

    # Error Analysis before applying optimal threshold
    error_analysis_results_before = error_analysis(train_x, train_y, model)
    error_analysis_results_before.to_csv("./output/error_analysis_before_threshold.csv", index=False)

    # Find the optimal threshold using bayesian optimization
    optimal_threshold = find_optimal_threshold(train_x, train_y, model)
    # # Or Set Optimal_threshold
    # optimal_threshold = 0.5056973148159999

    scorer = threshold_scorer(optimal_threshold)
    accuracy_scores_after_threshold = cross_val_score(model, train_x, train_y, scoring=scorer, n_jobs=-1, cv=10)
    print(f"Cross-Validation Accuracy Scores After Threshold: {accuracy_scores_after_threshold}")
    print(f"Average Cross-Validated Accuracy After Threshold: {np.mean(accuracy_scores_after_threshold):.4f}")

    y_probs = model.predict_proba(train_x)[:, 1]
    # Error Analysis after applying optimal threshold
    y_pred = (y_probs >= optimal_threshold).astype(int)
    error_analysis_results_after = error_analysis(train_x, train_y, model, threshold=optimal_threshold)
    error_analysis_results_after.to_csv("./output/error_analysis_after_threshold.csv", index=False)

    # Output model importances
    importance_df = pd.DataFrame({"Feature": train_x.columns, "Importance": model.feature_importances_}).sort_values(
        by="Importance", ascending=False
    )

    importance_df.to_csv("./output/feature-importance.txt", index=False, sep="\t")

    # Create a Submission with the New Model
    sample_submission_df = load_sample_submission(sample_submission_path)

    test_x = test_df.drop(columns=["id"])
    y_probs = model.predict_proba(test_x)[:, 1]
    y_pred = (y_probs >= optimal_threshold).astype(int)

    timestamp = datetime.now().strftime("%m_%d_%Y_%H-%M")
    submission_file_name = f"submission_{timestamp}.csv"
    submission_path = os.path.join(submission_dir, submission_file_name)

    update_submission_structure(sample_submission_df, y_pred)
    save_submission(sample_submission_df, submission_path)

    save_lgbm_model(model, timestamp=timestamp)
