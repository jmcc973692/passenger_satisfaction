import json
import os
import warnings

import lightgbm as lgb
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Function to save hyperparameters
def save_hyperparameters(algorithm_name, best_params):
    directory = "./hyperparameters"
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the hyperparameters to a JSON file
    filepath = os.path.join(directory, f"{algorithm_name}.json")
    with open(filepath, "w") as file:
        json.dump(best_params, file)


def xgb_objective(space):
    model = xgb.XGBClassifier(
        n_estimators=int(space["n_estimators"]),
        learning_rate=space["learning_rate"],
        max_depth=int(space["max_depth"]),
        min_child_weight=space["min_child_weight"],
        gamma=space["gamma"],
        subsample=space["subsample"],
        colsample_bytree=space["colsample_bytree"],
        # We won't include booster here as we're not considering 'dart'
        # booster=space["booster"]  # removed this line
    )
    accuracy = cross_val_score(model, space["X"], space["y"], cv=5).mean()
    return {"loss": -accuracy, "status": STATUS_OK}


def rf_objective(space):
    model = RandomForestClassifier(
        n_estimators=int(space["n_estimators"]),
        max_depth=int(space["max_depth"]),
        min_samples_split=int(space["min_samples_split"]),
        min_samples_leaf=int(space["min_samples_leaf"]),
        max_features=space["max_features"],
    )
    accuracy = cross_val_score(model, space["X"], space["y"], cv=5).mean()
    return {"loss": -accuracy, "status": STATUS_OK}


def lgbm_objective(space):
    model = lgb.LGBMClassifier(
        n_estimators=int(space["n_estimators"]),
        learning_rate=space["learning_rate"],
        max_depth=int(space["max_depth"]),
        num_leaves=int(space["num_leaves"]),
        min_child_samples=int(space["min_child_samples"]),
        subsample=space["subsample"],
        colsample_bytree=space["colsample_bytree"],
    )
    accuracy = cross_val_score(model, space["X"], space["y"], cv=5).mean()
    return {"loss": -accuracy, "status": STATUS_OK}


def tune_xgb_parameters(X, y, max_evals=100):
    space = {
        "n_estimators": hp.choice("n_estimators", range(50, 1000, 50)),  # Expanded to 1000
        "learning_rate": hp.quniform("learning_rate", 0.001, 0.5, 0.001),  # Expanded range & granularity
        "max_depth": hp.choice("max_depth", range(2, 15)),  # Expanded range
        "min_child_weight": hp.choice("min_child_weight", range(1, 8)),  # Expanded range
        "gamma": hp.quniform("gamma", 0, 0.8, 0.01),  # Expanded upper limit
        "subsample": hp.quniform("subsample", 0.4, 1, 0.05),  # Expanded range
        "colsample_bytree": hp.quniform("colsample_bytree", 0.4, 1, 0.05),  # Expanded range
        "X": X,
        "y": y,
    }

    trials = Trials()
    best_params = fmin(fn=xgb_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    # Convert the indices of 'n_estimators', 'max_depth', and 'min_child_weight' to actual values
    best_params["n_estimators"] = [i for i in range(50, 1000, 50)][best_params["n_estimators"]]
    best_params["max_depth"] = list(range(2, 15))[best_params["max_depth"]]
    best_params["min_child_weight"] = list(range(1, 8))[best_params["min_child_weight"]]
    save_hyperparameters("xgb", best_params)
    return best_params


def tune_rf_parameters(X, y, max_evals=100):
    space = {
        "n_estimators": hp.choice("n_estimators", range(50, 500, 50)),
        "max_depth": hp.choice("max_depth", range(3, 15)),
        "min_samples_split": hp.choice("min_samples_split", range(2, 20)),
        "min_samples_leaf": hp.choice("min_samples_leaf", range(1, 10)),
        "max_features": hp.choice("max_features", ["sqrt", "log2"]),
        "X": X,
        "y": y,
    }

    trials = Trials()
    best_params = fmin(fn=rf_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_params["n_estimators"] = [i for i in range(50, 500, 50)][best_params["n_estimators"]]
    best_params["max_depth"] = list(range(3, 15))[best_params["max_depth"]]
    best_params["min_samples_split"] = list(range(2, 20))[best_params["min_samples_split"]]
    best_params["min_samples_leaf"] = list(range(1, 10))[best_params["min_samples_leaf"]]
    best_params["max_features"] = ["auto", "sqrt", "log2"][best_params["max_features"]]
    save_hyperparameters("rf", best_params)
    return best_params


def tune_lgbm_parameters(X, y, max_evals=100):
    space = {
        "n_estimators": hp.choice("n_estimators", range(50, 500, 50)),
        "learning_rate": hp.quniform("learning_rate", 0.01, 0.2, 0.01),
        "max_depth": hp.choice("max_depth", range(3, 15)),
        "num_leaves": hp.choice("num_leaves", range(20, 150)),
        "min_child_samples": hp.choice("min_child_samples", range(5, 50)),
        "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
        "X": X,
        "y": y,
    }

    trials = Trials()
    best_params = fmin(fn=lgbm_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_params["n_estimators"] = [i for i in range(50, 500, 50)][best_params["n_estimators"]]
    best_params["max_depth"] = list(range(3, 15))[best_params["max_depth"]]
    best_params["num_leaves"] = list(range(20, 150))[best_params["num_leaves"]]
    best_params["min_child_samples"] = list(range(5, 50))[best_params["min_child_samples"]]
    save_hyperparameters("lgbm", best_params)
    return best_params
