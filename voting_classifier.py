import json
import os
import warnings
from datetime import datetime

import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score

from src.data_handling import drop_low_importance_features, keep_best_features_only, prepare_data
from src.feature_engineering import perform_feature_engineering
from src.submission import load_sample_submission, save_submission, update_submission_structure

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


def get_hyperparameters(algorithm):
    file_path = f"./hyperparameters/{algorithm}.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"No hyperparameters found for {algorithm}")


def main(train_path, test_path, sample_submission_path, submission_dir):
    train_df, test_df = prepare_data(train_path, test_path)

    # Feature Engineering Steps
    train_df = perform_feature_engineering(train_df)
    test_df = perform_feature_engineering(test_df)

    best_features = [
        "Age_x_Type_of_Travel_Personal",
        "Baggage_Handling",
        "Check-In_Service",
        "Class_Economy",
        "Class_Economy_Plus_x_Cleanliness",
        "Cleanliness",
        "Convenience_of_Departure/Arrival_Time_",
        "Customer_Type_Non-Loyal_Customer_x_On-Board_Service",
        "Departure_Delay_in_Minutes",
        "Ease_of_Online_booking",
        "Flight_Delay_Difference_Made_Up_Time",
        "Gate_Location",
        "Inflight_Entertainment",
        "Inflight_Service",
        "Inflight_Wifi_Service",
        "Leg_Room",
        "On-Board_Service",
        "Online_Boarding",
        "Online_Boarding_x_Ease_of_Online_booking",
        "Seat_Comfort",
        "Seat_Comfort_x_Leg_Room",
        "Type_of_Travel_Personal",
        "Age",
        "Flight_Distance_bins_Long-Haul",
        "Flight_Distance_bins_Medium-Haul",
    ]

    train_df = keep_best_features_only(train_df, best_features=best_features)
    test_df = keep_best_features_only(test_df, best_features=best_features)

    # Drop the ID and target before training
    train_x = train_df.drop(columns=["Satisfaction_Rating", "id"])
    train_y = train_df["Satisfaction_Rating"]

    # Load Hyperparameters
    xgb_params = get_hyperparameters("xgb")
    rf_params = get_hyperparameters("rf")
    lgbm_params = get_hyperparameters("lgbm")

    # Create Classifier Instances
    xgb_clf = xgb.XGBClassifier(**xgb_params)
    rf_clf = RandomForestClassifier(**rf_params)
    lgbm_clf = LGBMClassifier(**lgbm_params)

    # Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[("xgb", xgb_clf), ("rf", rf_clf), ("lgbm", lgbm_clf)], voting="hard", weights=[1.0, 1.0, 1.5]
    )

    # Train the Voting Classifier
    voting_clf.fit(train_x, train_y)

    scores = cross_val_score(voting_clf, train_x, train_y, cv=5)
    print("Cross-validated scores for voting classifier:", scores)
    print("Average score for voting classifier:", scores.mean())

    # Evaluate model accuracy on training data
    accuracy = (voting_clf.predict(train_x) == train_y).sum() / len(train_y)
    print(f"Accuracy on training data for voting classifier: {accuracy:.4f}")

    sample_submission_df = load_sample_submission(sample_submission_path)

    test_x = test_df.drop(columns=["id"])
    y_pred = voting_clf.predict(test_x)

    timestamp = datetime.now().strftime("%m_%d_%Y_%H-%M")
    submission_file_name = f"submission_voting_{timestamp}.csv"
    submission_path = os.path.join(submission_dir, submission_file_name)

    update_submission_structure(sample_submission_df, y_pred)
    save_submission(sample_submission_df, submission_path)


if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    sample_submission_path = "./input/sample_submission.csv"
    submission_dir = "./submission"

    main(train_path, test_path, sample_submission_path, submission_dir)
