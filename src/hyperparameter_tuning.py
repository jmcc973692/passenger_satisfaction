import json
import os
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

import lightgbm as lgb
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


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
        booster=space["booster"],
        n_estimators=int(space["n_estimators"]),
        learning_rate=space["learning_rate"],
        max_depth=int(space["max_depth"]),
        min_child_weight=space["min_child_weight"],
        gamma=space["gamma"],
        subsample=space["subsample"],
        colsample_bytree=space["colsample_bytree"],
        colsample_bylevel=space["colsample_bylevel"],
        colsample_bynode=space["colsample_bynode"],
        reg_alpha=space["reg_alpha"],
        reg_lambda=space["reg_lambda"],
        scale_pos_weight=space["scale_pos_weight"],
        objective=space["objective"],
        eval_metric=space["eval_metric"],
        grow_policy=space["grow_policy"],
    )

    model.fit(space["X"], space["y"])
    accuracy = cross_val_score(model, space["X"], space["y"], cv=5, scoring="accuracy").mean()
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


def tune_xgb_parameters(X, y, max_evals=200):  # Increased max_evals
    space = {
        "learning_rate": hp.loguniform("learning_rate", -5, 0),  # A log uniform from ~0.007 to 1 might be sensible
        "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),  # You can adjust the range based on dataset size
        "max_depth": hp.choice("max_depth", list(range(3, 15))),  # Common choice for depth
        "gamma": hp.quniform("gamma", 0, 0.5, 0.01),  # A range of 0 to 0.5 by 0.01
        "subsample": hp.quniform("subsample", 0.5, 1, 0.05),  # Values from 0.5 to 1 with a step of 0.05
        "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),  # Similar to above
        "colsample_bylevel": hp.quniform("colsample_bylevel", 0.5, 1, 0.05),
        "colsample_bynode": hp.quniform("colsample_bynode", 0.5, 1, 0.05),
        "reg_lambda": hp.loguniform("reg_lambda", -5, 2),  # Log uniform for regularization
        "reg_alpha": hp.loguniform("reg_alpha", -5, 2),
        "scale_pos_weight": hp.loguniform(
            "scale_pos_weight", 0, 2
        ),  # Log uniform, especially useful for imbalanced datasets
        "grow_policy": hp.choice("grow_policy", ["depthwise", "lossguide"]),
        "n_estimators": hp.quniform("n_estimators", 50, 1000, 1),
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "error",
        "X": X,
        "y": y,
    }

    trials = Trials()
    best_params = fmin(fn=xgb_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    # Map back the integer values for certain parameters
    best_params["min_child_weight"] = int(best_params["min_child_weight"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["gamma"] = float(best_params["gamma"])
    best_params["subsample"] = float(best_params["subsample"])
    best_params["colsample_bytree"] = float(best_params["colsample_bytree"])
    best_params["colsample_bylevel"] = float(best_params["colsample_bylevel"])
    best_params["colsample_bynode"] = float(best_params["colsample_bynode"])
    best_params["n_estimators"] = int(best_params["n_estimators"])
    grow_policy_choices = ["depthwise", "lossguide"]
    best_params["grow_policy"] = grow_policy_choices[best_params["grow_policy"]]
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "error"

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


def tune_lgbm_parameters(X, y, max_evals=150):
    space = {
        "n_estimators": hp.choice("n_estimators", range(200, 400, 10)),  # Narrowed around 300
        "learning_rate": hp.quniform("learning_rate", 0.01, 0.05, 0.005),  # Narrowed around 0.03
        "max_depth": hp.choice("max_depth", range(10, 20)),  # Narrowed around 14
        "num_leaves": hp.choice("num_leaves", range(100, 160, 5)),  # Narrowed around 132
        "min_child_samples": hp.choice("min_child_samples", range(2, 10)),  # Narrowed around 6
        "subsample": hp.quniform("subsample", 0.6, 0.9, 0.05),  # Narrowed around 0.75
        "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 0.6, 0.01),  # Narrowed around 0.55
        "X": X,
        "y": y,
    }

    trials = Trials()
    best_params = fmin(fn=lgbm_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_params["n_estimators"] = range(200, 400, 10)[best_params["n_estimators"]]
    best_params["max_depth"] = range(10, 20)[best_params["max_depth"]]
    best_params["num_leaves"] = range(100, 160, 5)[best_params["num_leaves"]]
    best_params["min_child_samples"] = range(2, 10)[best_params["min_child_samples"]]

    save_hyperparameters("lgbm", best_params)
    return best_params
