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
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, cross_val_score

from src.adversarial_validation import get_adversarial_val_set
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


def save_lgbm_model(model, timestamp):
    model_file_name = f"lgbm_model_{timestamp}.pkl"
    model_path = os.path.join("./output/models", model_file_name)
    joblib.dump(model, model_path)


def select_high_confidence_predictions(predictions, threshold=0.9):
    high_confidence_mask = predictions > threshold
    return high_confidence_mask


def add_pseudo_labels(train_data, test_data, train_labels, predictions, high_confidence_mask):
    pseudo_labeled_data = test_data[high_confidence_mask]
    pseudo_labels = (predictions[high_confidence_mask] > 0.5).astype(int)  # Convert probabilities to class labels

    augmented_train_data = pd.concat([train_data, pseudo_labeled_data])
    augmented_train_labels = pd.concat([train_labels, pd.Series(pseudo_labels)])

    return augmented_train_data, augmented_train_labels


if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    sample_submission_path = "./input/sample_submission.csv"
    submission_dir = "./submission"

    train_df, test_df = prepare_data(train_path, test_path)

    # Feature Engineering Steps
    train_df = perform_feature_engineering_lgbm(train_df)
    test_df = perform_feature_engineering_lgbm(test_df)

    # Drop the ID and target before training
    train_df = train_df.drop(columns=["id"])
    test_df = test_df.drop(columns=["id"])

    train_x = train_df.drop(columns=["Satisfaction_Rating"])
    train_y = train_df["Satisfaction_Rating"]

    # Create the same StratifiedKFold from the hyperparameter tuning
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    # Cross-Validation before applying optimal threshold
    model = train_lgbm(train_x=train_x, train_y=train_y)
    scores = cross_val_score(model, train_x, train_y, cv=rkf, n_jobs=-1, scoring="accuracy")
    print(f"Initial Cross-Validation Accuracy Scores: {scores}")
    print(f"Initial Average cross-validated accuracy: {np.mean(scores):.4f}")

    accuracy = (model.predict(train_x) == train_y).sum() / len(train_y)
    print(f"Initial Accuracy on Full Training Data: {accuracy:.4f}")

    # Error Analysis before applying optimal threshold
    error_analysis_results_before = error_analysis(train_x, train_y, model)

    # Output model importances
    importance_df = pd.DataFrame({"Feature": train_x.columns, "Importance": model.feature_importances_}).sort_values(
        by="Importance", ascending=False
    )
    importance_df.to_csv("./output/feature-importance.txt", index=False, sep="\t")

    ### OPTIMAL THRESHOLD
    # Find the optimal threshold using bayesian optimization
    # optimal_threshold = find_optimal_threshold(train_x, train_y, model)
    # # Or Set Optimal_threshold
    # # optimal_threshold = 0.5026910287740821
    optimal_threshold = 0.5

    # scorer = threshold_scorer(optimal_threshold)
    # accuracy_scores_after_threshold = cross_val_score(model, train_x, train_y, scoring=scorer, n_jobs=-1, cv=rkf)
    # print(f"Cross-Validation Accuracy Scores After Threshold: {accuracy_scores_after_threshold}")
    # print(f"Average Cross-Validated Accuracy After Threshold: {np.mean(accuracy_scores_after_threshold):.4f}")

    y_probs = model.predict_proba(train_x)[:, 1]
    # Error Analysis after applying optimal threshold
    y_pred = (y_probs >= optimal_threshold).astype(int)
    accuracy_after = (y_pred == train_y).sum() / len(train_y)
    print(f"Accuracy on training data after threshold: {accuracy_after:.4f}")

    error_analysis_results_after = error_analysis(train_x, train_y, model, threshold=optimal_threshold)

    # Create a Submission with the New Model
    sample_submission_df = load_sample_submission(sample_submission_path)

    y_probs = model.predict_proba(test_df)[:, 1]
    y_pred = (y_probs >= optimal_threshold).astype(int)

    timestamp = datetime.now().strftime("%m_%d_%Y_%H-%M")
    submission_file_name = f"submission_{timestamp}.csv"
    submission_path = os.path.join(submission_dir, submission_file_name)

    update_submission_structure(sample_submission_df, y_pred)
    save_submission(sample_submission_df, submission_path)

    save_lgbm_model(model, timestamp=timestamp)
