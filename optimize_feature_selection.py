import json
import os
import warnings
from functools import partial

import numpy as np
import pandas as pd
from hyperopt import STATUS_FAIL, STATUS_OK, fmin, hp, tpe
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.data_handling import drop_low_importance_features, prepare_data
from src.feature_engineering import perform_feature_engineering

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

evaluated_combinations = set()

# Default model parameters for XGBClassifier
DEFAULT_MODEL_PARAMS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "learning_rate": 0.05,  # Reduced learning rate
    "n_estimators": 50,  # Reduced number of trees
    "max_depth": 4,  # Reduced depth
    "min_child_weight": 5,  # Increased minimum child weight
    "gamma": 0.2,  # Increase if you'd like further regularization
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "colsample_bynode": 0.7,
    "alpha": 0.2,  # Increased L1 regularization
    "lambda": 2,  # Increased L2 regularization
    "eval_metric": "auc",
    "random_state": 42,
}


def evaluate_model(features, train_df, model_params=DEFAULT_MODEL_PARAMS):
    model = XGBClassifier(**model_params)

    # Filter columns based on selected features
    X = train_df[features]
    y = train_df["Satisfaction_Rating"]

    # Split into training and validation for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)])

    # Using cross-validation for evaluation
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

    return np.mean(scores)


def objective(params):
    # Extracting features
    features = [k for k, v in params.items() if v]

    # Ensure we're not re-evaluating
    feature_tuple = tuple(sorted(features))
    if feature_tuple in evaluated_combinations:
        return {"loss": np.inf, "status": STATUS_FAIL}

    score = -evaluate_model(features, train_df)

    # Write to output
    with open("./output/optimize_feature_results.txt", "a") as f:
        f.write(f"Features: {features}\n")
        f.write(f"Score: {-score}\n")
        f.write("-" * 40 + "\n")

    evaluated_combinations.add(feature_tuple)
    return {"loss": score, "status": STATUS_OK}


def optimize_feature_selection(train_df):
    space = {
        "Online_Boarding": hp.choice("Online_Boarding", [True, False]),
        "Type_of_Travel_Personal": hp.choice("Type_of_Travel_Personal", [True, False]),
        "Inflight_Wifi_Service": hp.choice("Inflight_Wifi_Service", [True, False]),
        "Customer_Type_Non-Loyal Customer": hp.choice("Customer_Type_Non-Loyal Customer", [True, False]),
        "Wifi_OnlineBoarding": hp.choice("Wifi_OnlineBoarding", [True, False]),
        "Class_Economy": hp.choice("Class_Economy", [True, False]),
        "Inflight_Entertainment_x_On-Board_Service": hp.choice(
            "Inflight_Entertainment_x_On-Board_Service", [True, False]
        ),
        "Seat_Comfort_x_Leg_Room": hp.choice("Seat_Comfort_x_Leg_Room", [True, False]),
        "Seat_Comfort": hp.choice("Seat_Comfort", [True, False]),
        "Gate_Location": hp.choice("Gate_Location", [True, False]),
        "Cleanliness": hp.choice("Cleanliness", [True, False]),
        "Check-In_Service": hp.choice("Check-In_Service", [True, False]),
        "Baggage_Handling": hp.choice("Baggage_Handling", [True, False]),
        "Ease_of_Online_booking": hp.choice("Ease_of_Online_booking", [True, False]),
        "OnBoard_InflightService": hp.choice("OnBoard_InflightService", [True, False]),
        "Flight_Distance_bins": hp.choice("Flight_Distance_bins", [True, False]),
        "Leg_Room": hp.choice("Leg_Room", [True, False]),
        "Gender_TypeOfTravel": hp.choice("Gender_TypeOfTravel", [True, False]),
        "Inflight_Entertainment": hp.choice("Inflight_Entertainment", [True, False]),
        "Inflight_Service": hp.choice("Inflight_Service", [True, False]),
        "Class_Economy Plus": hp.choice("Class_Economy Plus", [True, False]),
        "Online_Boarding_x_Ease_of_Online_booking": hp.choice(
            "Online_Boarding_x_Ease_of_Online_booking", [True, False]
        ),
        "Convenience_of_Departure/Arrival_Time_": hp.choice("Convenience_of_Departure/Arrival_Time_", [True, False]),
        "Age_bins": hp.choice("Age_bins", [True, False]),
        "Gender_Male": hp.choice("Gender_Male", [True, False]),
        "Age": hp.choice("Age", [True, False]),
        "Arrival_Delay_in_Minutes^2": hp.choice("Arrival_Delay_in_Minutes^2", [True, False]),
        "Age^3": hp.choice("Age^3", [True, False]),
        "Arrival_Delay_in_Minutes": hp.choice("Arrival_Delay_in_Minutes", [True, False]),
        "Food_SeatComfort": hp.choice("Food_SeatComfort", [True, False]),
        "Arrival_Delay_in_Minutes^3": hp.choice("Arrival_Delay_in_Minutes^3", [True, False]),
        "FlightDistance_Class": hp.choice("FlightDistance_Class", [True, False]),
        "Flight_Distance^2": hp.choice("Flight_Distance^2", [True, False]),
        "Age_x_Flight_Distance": hp.choice("Age_x_Flight_Distance", [True, False]),
        "On-Board_Service": hp.choice("On-Board_Service", [True, False]),
        "Flight_Distance^3": hp.choice("Flight_Distance^3", [True, False]),
        "Food_and_Drink": hp.choice("Food_and_Drink", [True, False]),
        "Flight_Distance": hp.choice("Flight_Distance", [True, False]),
        "Age^2": hp.choice("Age^2", [True, False]),
        "Departure_Delay_in_Minutes_x_Arrival_Delay_in_Minutes": hp.choice(
            "Departure_Delay_in_Minutes_x_Arrival_Delay_in_Minutes", [True, False]
        ),
        "Departure_Delay_in_Minutes": hp.choice("Departure_Delay_in_Minutes", [True, False]),
        "Departure_Delay_in_Minutes^3": hp.choice("Departure_Delay_in_Minutes^3", [True, False]),
        "Departure_Delay_in_Minutes^2": hp.choice("Departure_Delay_in_Minutes^2", [True, False]),
        "Experienced_Delay": hp.choice("Experienced_Delay", [True, False]),
    }

    optimized = fmin(
        fn=partial(objective),
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        rstate=np.random.default_rng(42),
    )

    best_features = [feature for feature, use in optimized.items() if use]
    print(f"Best features to use are: {best_features}")


if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    train_df, test_df = prepare_data(train_path, test_path)

    # Feature Engineering Steps
    train_df = perform_feature_engineering(train_df)
    test_df = perform_feature_engineering(test_df)

    optimize_feature_selection(train_df)
