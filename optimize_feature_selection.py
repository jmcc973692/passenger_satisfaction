import json
import os
import warnings
from functools import partial

import numpy as np
import pandas as pd
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, pyll, rand, space_eval, tpe
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.data_handling import prepare_data
from src.feature_engineering import perform_feature_engineering

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

evaluated_combinations = set()

# # Default model parameters for XGBClassifier
# DEFAULT_MODEL_PARAMS_XGB = {
#     "objective": "binary:logistic",
#     "booster": "gbtree",
#     "learning_rate": 0.03,  # More conservative learning rate
#     "n_estimators": 100,  # Increase number of trees
#     "max_depth": 6,  # Allow deeper trees for interactions
#     "min_child_weight": 3,  # Mid-way value
#     "gamma": 0.1,  # Regularization on leaf nodes
#     "subsample": 0.8,  # Randomly sample 80% of the training data on each round
#     "colsample_bytree": 0.8,  # Randomly sample 80% of features on each round
#     "colsample_bylevel": 0.8,  # Randomly sample 80% of features on each level
#     "colsample_bynode": 0.8,  # Randomly sample 80% of features for each node split
#     "alpha": 0.1,  # L1 regularization
#     "lambda": 1,  # L2 regularization
# }

# DEFAULT_MODEL_PARAMS_XGB = {
#     "n_estimators": 100,  # Number of boosting rounds. You can set it to a higher value and rely on early_stopping_rounds to find the optimal number.
#     "learning_rate": 0.05,  # Makes the optimization more robust and better generalization. Default is 0.3.
#     "max_depth": 5,  # Maximum depth of a tree. Increasing this will make the model more complex and likely to overfit.
#     "min_child_weight": 1,  # Minimum sum of instance weight (hessian) needed in a child. Use it to control overfitting.
#     "subsample": 0.8,  # Fraction of training data to train XGBoost. Setting it to 0.5 means XGBoost randomly collects half of the data instances to grow trees.
#     "colsample_bytree": 0.8,  # Fraction of features to be randomly sampled for building each tree.
#     "reg_alpha": 0.01,  # L1 regularization term on weights. Can be used in case of high dimensionality to make the algorithm run faster.
#     "reg_lambda": 1,  # L2 regularization term on weights.
#     "gamma": 0.1,  # Minimum loss reduction required to make a further partition on a leaf node. It acts as regularization on the tree.
#     "scale_pos_weight": 1,  # Controls the balance of positive and negative weights. Useful for imbalanced classes.
#     "objective": "binary:logistic",  # Objective function. Assuming a binary classification problem. Change it if necessary.
# }

# Parameters from leaderboard best model
DEFAULT_MODEL_PARAMS_XGB = {
    "objective": "binary:logistic",
    "subsample": 0.7,
    "n_estimators": 200,
    "min_child_weight": 1,
    "max_depth": 8,
    "learning_rate": 0.05,
    "gamma": 0,
    "colsample_bytree": 1.0,
}

# DEFAULT_MODEL_PARAMS_LGBM = {
#     "colsample_bytree": 0.55,
#     "learning_rate": 0.03,
#     "max_depth": 14,
#     "min_child_samples": 6,
#     "n_estimators": 300,
#     "num_leaves": 132,
#     "subsample": 0.75,
# }

DEFAULT_MODEL_PARAMS_LGBM = {
    "bagging_freq": 2,
    "colsample_bytree": 0.6986396042821974,
    "learning_rate": 0.01319766004630036,
    "max_depth": 21,
    "min_child_samples": 8,
    "n_estimators": 900,
    "num_leaves": 109,
    "reg_alpha": 0.00011787116004744925,
    "reg_lambda": 0.06644285765092664,
    "subsample": 0.783851348243847,
}


def translate_selections(params):
    if params["Age_Selection"] == "numerical":
        params["Age"] = True
    elif params["Age_Selection"] == "bins":
        params["Age_bins_Young_Adult"] = True
        params["Age_bins_Adult"] = True
        params["Age_bins_Senior"] = True
    elif params["Age_Selection"] == "squared":
        params["Age"] = True
        params["Age^2"] = True
    elif params["Age_Selection"] == "cubed":
        params["Age"] = True
        params["Age^2"] = True
        params["Age^3"] = True
    else:
        pass
    del params["Age_Selection"]

    if params["Flight_Distance_Selection"] == "numerical":
        params["Flight_Distance"] = True
    elif params["Flight_Distance_Selection"] == "bins":
        params["Flight_Distance_bins_Long-Haul"] = True
        params["Flight_Distance_bins_Medium-Haul"] = True
    elif params["Flight_Distance_Selection"] == "squared":
        params["Flight_Distance"] = True
        params["Flight_Distance^2"] = True
    elif params["Flight_Distance_Selection"] == "cubed":
        params["Flight_Distance"] = True
        params["Flight_Distance^2"] = True
        params["Flight_Distance^3"] = True
    else:
        pass
    del params["Flight_Distance_Selection"]

    if params["Delay_Selection"] == "numerical":
        params["Departure_Delay_in_Minutes"] = True
        params["Arrival_Delay_in_Minutes"] = True
    elif params["Delay_Selection"] == "total":
        params["Total_Delay_Minutes"] = True
    elif params["Delay_Selection"] == "categorical":
        params["Experienced_Delay"] = True
    elif params["Delay_Selection"] == "squared":
        params["Departure_Delay_in_Minutes"] = True
        params["Arrival_Delay_in_Minutes"] = True
        params["Departure_Delay_in_Minutes^2"] = True
        params["Arrival_Delay_in_Minutes^2"] = True
    elif params["Delay_Selection"] == "cubed":
        params["Departure_Delay_in_Minutes"] = True
        params["Arrival_Delay_in_Minutes"] = True
        params["Departure_Delay_in_Minutes^2"] = True
        params["Arrival_Delay_in_Minutes^2"] = True
        params["Departure_Delay_in_Minutes^3"] = True
        params["Arrival_Delay_in_Minutes^3"] = True
    else:
        pass
    del params["Delay_Selection"]

    if params["Delay_Difference_Selection"] == "numerical":
        params["Flight_Delay_Difference_Minutes"] = True
    elif params["Delay_Difference_Selection"] == "bins":
        params["Flight_Delay_Difference_Made_Up_Time"] = True
        params["Flight_Delay_Difference_Lost_Time"] = True
    else:
        pass
    del params["Delay_Difference_Selection"]

    return params


def objective_lgbm(params, train_df, model_params=DEFAULT_MODEL_PARAMS_LGBM):
    # Extracting features
    params = translate_selections(params)
    features = [k for k, v in params.items() if v]
    model = LGBMClassifier(**model_params)

    x = train_df[features]
    y = train_df["Satisfaction_Rating"]

    model.fit(x, y)
    score = np.mean(cross_val_score(model, x, y, cv=5, scoring="accuracy"))

    # Write to output
    with open("./output/optimize_feature_results.txt", "a") as f:
        f.write(f"Features: {features}\n")
        f.write(f"Score: {-score}\n")
        f.write("-" * 40 + "\n")

    return {"loss": -score, "status": STATUS_OK}


def objective_xgb(params, train_df, model_params=DEFAULT_MODEL_PARAMS_XGB):
    # Extracting features
    params = translate_selections(params)
    features = [k for k, v in params.items() if v]
    model = XGBClassifier(**model_params)

    x = train_df[features]
    y = train_df["Satisfaction_Rating"]

    model.fit(x, y)
    score = np.mean(cross_val_score(model, x, y, cv=5, scoring="accuracy"))

    # Write to output
    with open("./output/optimize_feature_results.txt", "a") as f:
        f.write(f"Features: {features}\n")
        f.write(f"Score: {-score}\n")
        f.write("-" * 40 + "\n")

    return {"loss": -score, "status": STATUS_OK}


def optimize_feature_selection(train_df):
    space = {
        "Age_Selection": hp.choice("Age_Selection", ["numerical", "bins", "squared", "cubed", "no_selection"]),
        "Flight_Distance_Selection": hp.choice(
            "Flight_Distance_Selection", ["numerical", "bins", "squared", "cubed", "no_selection"]
        ),
        "Delay_Selection": hp.choice(
            "Delay_Selection", ["numerical", "total", "categorical", "squared", "cubed", "no_selection"]
        ),
        "Delay_Difference_Selection": hp.choice("Delay_Difference_Selection", ["numerical", "bins", "no_selection"]),
        "Ease_of_Online_booking": hp.choice("Ease_of_Online_booking", [False, True]),
        "Convenience_of_Departure/Arrival_Time_": hp.choice("Convenience_of_Departure/Arrival_Time_", [False, True]),
        "Baggage_Handling": hp.choice("Baggage_Handling", [False, True]),
        "Check-In_Service": hp.choice("Check-In_Service", [False, True]),
        "Gate_Location": hp.choice("Gate_Location", [False, True]),
        "Online_Boarding": hp.choice("Online_Boarding", [False, True]),
        "Inflight_Wifi_Service": hp.choice("Inflight_Wifi_Service", [False, True]),
        "Food_and_Drink": hp.choice("Food_and_Drink", [False, True]),
        "Seat_Comfort": hp.choice("Seat_Comfort", [False, True]),
        "Inflight_Entertainment": hp.choice("Inflight_Entertainment", [False, True]),
        "On-Board_Service": hp.choice("On-Board_Service", [False, True]),
        "Leg_Room": hp.choice("Leg_Room", [False, True]),
        "Inflight_Service": hp.choice("Inflight_Service", [False, True]),
        "Cleanliness": hp.choice("Cleanliness", [False, True]),
        "Gender_Male": hp.choice("Gender_Male", [False, True]),
        "Type_of_Travel_Personal": hp.choice("Type_of_Travel_Personal", [False, True]),
        "Class_Economy": hp.choice("Class_Economy", [False, True]),
        "Class_Economy_Plus": hp.choice("Class_Economy_Plus", [False, True]),
        "Online_Boarding_x_Ease_of_Online_booking": hp.choice(
            "Online_Boarding_x_Ease_of_Online_booking", [False, True]
        ),
        "Seat_Comfort_x_Leg_Room": hp.choice("Seat_Comfort_x_Leg_Room", [False, True]),
        "Inflight_Entertainment_x_Flight_Distance": hp.choice(
            "Inflight_Entertainment_x_Flight_Distance", [False, True]
        ),
        "Food_and_Drink_x_Flight_Distance": hp.choice("Food_and_Drink_x_Flight_Distance", [False, True]),
        "Age_x_Type_of_Travel_Personal": hp.choice("Age_x_Type_of_Travel_Personal", [False, True]),
        "Gender_Male_x_Inflight_Wifi_Service": hp.choice("Gender_Male_x_Inflight_Wifi_Service", [False, True]),
        "Departure_Delay_in_Minutes_x_Arrival_Delay_in_Minutes": hp.choice(
            "Departure_Delay_in_Minutes_x_Arrival_Delay_in_Minutes", [False, True]
        ),
        "Age_x_Inflight_Entertainment": hp.choice("Age_x_Inflight_Entertainment", [False, True]),
        "Class_Economy_x_Cleanliness": hp.choice("Class_Economy_x_Cleanliness", [False, True]),
        "Class_Economy_Plus_x_Cleanliness": hp.choice("Class_Economy_Plus_x_Cleanliness", [False, True]),
        "Customer_Type_Non-Loyal_Customer_x_On-Board_Service": hp.choice(
            "Customer_Type_Non-Loyal_Customer_x_On-Board_Service", [False, True]
        ),
        "Class_Economy_x_Flight_Distance": hp.choice("Class_Economy_x_Flight_Distance", [False, True]),
        "Class_Economy_Plus_x_Flight_Distance": hp.choice("Class_Economy_Plus_x_Flight_Distance", [False, True]),
        "Service_Quality_Score": hp.choice("Service_Quality_Score", [False, True]),
        "Comfort_Score": hp.choice("Comfort_Score", [False, True]),
        "Convenience_Score": hp.choice("Convenience_Score", [False, True]),
        "diff_inflight_onboard_service": hp.choice("diff_inflight_onboard_service", [False, True]),
        "diff_seatcomfort_legroom": hp.choice("diff_seatcomfort_legroom", [False, True]),
        "diff_wifi_onlineboarding": hp.choice("diff_wifi_onlineboarding", [False, True]),
        "diff_food_cleanliness": hp.choice("diff_food_cleanliness", [False, True]),
    }

    trials = Trials()
    fmin(
        fn=partial(objective_lgbm, train_df=train_df),
        space=space,
        algo=rand.suggest,
        max_evals=250,
        rstate=np.random.default_rng(393),
        trials=trials,
    )

    fmin(
        fn=partial(objective_lgbm, train_df=train_df),
        space=space,
        algo=tpe.suggest,
        max_evals=750,
        rstate=np.random.default_rng(5564),
        trials=trials,
    )

    # Final Results
    best_features = space_eval(space, trials.argmin)
    best_features = translate_selections(best_features)
    best_features = [k for k, v in best_features.items() if v]
    print(f"Best features after optimization are: {best_features}")

    # Save results to a file in the ./output directory
    with open("./output/best_features.txt", "w") as file:
        formatted_features = ", ".join(f'"{feature}"' for feature in best_features)
        file.write(f"[{formatted_features}]")


if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    train_df, test_df = prepare_data(train_path, test_path)

    # Feature Engineering Steps
    train_df = perform_feature_engineering(train_df)
    test_df = perform_feature_engineering(test_df)

    optimize_feature_selection(train_df)
