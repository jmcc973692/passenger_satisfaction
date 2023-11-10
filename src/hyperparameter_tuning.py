import json
import os
import pickle
import warnings
from functools import partial

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

import lightgbm as lgb
import numpy as np
import torch.nn as nn
import torch.optim
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, rand, space_eval, tpe
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader, TensorDataset

from src.DynamicTabularNN import DynamicTabularNN
from src.TabularNN import TabularNN


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

    accuracy = cross_val_score(
        model, space["X"], space["y"], cv=5, scoring="accuracy"
    ).mean()
    return {"loss": -accuracy, "status": STATUS_OK}


def tune_xgb_parameters(X, y, max_evals=200):  # Increased max_evals
    space = {
        "learning_rate": hp.loguniform(
            "learning_rate", -5, 0
        ),  # A log uniform from ~0.007 to 1 might be sensible
        "min_child_weight": hp.quniform(
            "min_child_weight", 1, 10, 1
        ),  # You can adjust the range based on dataset size
        "max_depth": hp.choice(
            "max_depth", list(range(3, 15))
        ),  # Common choice for depth
        "gamma": hp.quniform("gamma", 0, 0.5, 0.01),  # A range of 0 to 0.5 by 0.01
        "subsample": hp.quniform(
            "subsample", 0.5, 1, 0.05
        ),  # Values from 0.5 to 1 with a step of 0.05
        "colsample_bytree": hp.quniform(
            "colsample_bytree", 0.5, 1, 0.05
        ),  # Similar to above
        "colsample_bylevel": hp.quniform("colsample_bylevel", 0.5, 1, 0.05),
        "colsample_bynode": hp.quniform("colsample_bynode", 0.5, 1, 0.05),
        "reg_lambda": hp.loguniform(
            "reg_lambda", -5, 2
        ),  # Log uniform for regularization
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
    best_params = fmin(
        fn=xgb_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )

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
    best_params = fmin(
        fn=rf_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )

    best_params["n_estimators"] = [i for i in range(50, 500, 50)][
        best_params["n_estimators"]
    ]
    best_params["max_depth"] = list(range(3, 15))[best_params["max_depth"]]
    best_params["min_samples_split"] = list(range(2, 20))[
        best_params["min_samples_split"]
    ]
    best_params["min_samples_leaf"] = list(range(1, 10))[
        best_params["min_samples_leaf"]
    ]
    best_params["max_features"] = ["auto", "sqrt", "log2"][best_params["max_features"]]
    save_hyperparameters("rf", best_params)
    return best_params


def lgbm_objective(space, X, y):
    model = lgb.LGBMClassifier(
        colsample_bytree=space["colsample_bytree"],
        learning_rate=space["learning_rate"],
        max_bin=int(space["max_bin"]),
        max_depth=int(space["max_depth"]),
        min_child_samples=int(space["min_child_samples"]),
        min_child_weight=space["min_child_weight"],
        min_split_gain=space["min_split_gain"],
        n_estimators=int(space["n_estimators"]),
        num_leaves=int(space["num_leaves"]),
        reg_alpha=space["reg_alpha"],
        reg_lambda=space["reg_lambda"],
        subsample=space["subsample"],
        subsample_freq=int(space["subsample_freq"]),
        verbose=-1,
    )
    accuracy = cross_val_score(model, X, y, cv=10, n_jobs=-1).mean()
    return {"loss": -accuracy, "status": STATUS_OK}


def tune_lgbm_parameters(X, y, max_evals=750):
    space = {
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.9),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.05)),
        "max_bin": hp.quniform("max_bin", 180, 300, 1),
        "max_depth": hp.quniform("max_depth", 10, 30, 1),
        "min_child_samples": hp.quniform("min_child_samples", 5, 50, 1),
        "min_child_weight": hp.loguniform(
            "min_child_weight", np.log(0.0001), np.log(0.01)
        ),
        "min_split_gain": hp.loguniform("min_split_gain", np.log(0.0001), np.log(0.01)),
        "n_estimators": hp.quniform("n_estimators", 800, 1500, 1),
        "num_leaves": hp.quniform("num_leaves", 80, 200, 1),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.00001), np.log(0.01)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.001), np.log(0.05)),
        "subsample": hp.uniform("subsample", 0.5, 0.9),
        "subsample_freq": hp.quniform("subsample_freq", 0, 7, 1),
    }
    trials = Trials()
    objective = partial(lgbm_objective, X=X, y=y)
    fmin(
        fn=objective,
        space=space,
        algo=rand.suggest,
        rstate=np.random.default_rng(3920),
        max_evals=150,
        trials=trials,
    )
    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        rstate=np.random.default_rng(964231),
        max_evals=max_evals,
        trials=trials,
    )

    best_params = space_eval(space, trials.argmin)

    # Now cast the necessary parameters to integers
    best_params["max_bin"] = int(best_params["max_bin"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_child_samples"] = int(best_params["min_child_samples"])
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["num_leaves"] = int(best_params["num_leaves"])
    best_params["subsample_freq"] = int(best_params["subsample_freq"])
    save_hyperparameters("lgbm", best_params)
    return best_params


def nn_objective(space, X_train, y_train, X_val, y_val, X_test, y_test, device):
    # Unpack the Space
    batch_size = space["batch_size"]
    dropout_rate = space["dropout"]
    factor = space["factor"]
    scheduler_patience = space["patience"]
    activation_functions = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "swish": nn.SiLU(),  # SiLU (Swish) was added in PyTorch 1.7.0 as nn.SiLU()
    }
    activation_func = activation_functions[space["activation"]]
    use_batch_norm = space["use_batch_norm"]

    # Create DataLoader Objects
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=10,
        persistent_workers=True,
    )

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=10,
        persistent_workers=True,
    )

    input_dim = 79
    layers = space["layers"]
    model = DynamicTabularNN(
        input_dim,
        dropout_rate=dropout_rate,
        layers=layers,
        activation_func=activation_func,
        use_batch_norm=use_batch_norm,
    )
    model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model.parameters(), space)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=scheduler_patience, verbose=True
    )

    # Early stopping parameters
    patience = 15  # number of epochs to wait for improvement before terminating
    best_val_loss = float("inf")
    epochs_no_improve = 0
    epochs = 1000

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}], Training Loss: {running_loss/len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(
                    device
                )

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}], Validation Loss: {val_loss/len(val_loader):.4f}")

        # Step the Scheduler
        scheduler.step(val_loss)

        # Check if the validation loss improved
        if best_val_loss - val_loss > 0.0002:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            epochs_no_improve = 0  # Reset the Counter
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early Stopping!")
            break

    model.load_state_dict(best_model_weights)
    with torch.no_grad():
        model.eval()
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        test_predictions = model(X_test)
        test_accuracy = compute_accuracy(test_predictions, y_test)

    # Save the results to a file after each trial
    save_results_to_file(space, test_accuracy, "./output/trials_results.txt")

    print(f"Test Accuracy - {test_accuracy:.4f}")
    return {"loss": -test_accuracy, "status": STATUS_OK}


def tune_nn_parameters(X, y, X_val, y_val, X_test, y_test, device):
    space = {
        "batch_size": hp.choice("batch_size", [64, 128, 256]),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-1)),
        "dropout": hp.choice(
            "dropout",
            [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
        ),
        "layers": hp.choice(
            "layers",
            [
                [384, 384, 384],
                [512, 384, 256],
                [192, 384, 192],
                [384, 192, 384],
                [128, 128, 128],
                [256, 128, 64],
                [128, 64, 32],
                [512, 256, 128],
                [384, 192, 96],
                [256, 128, 64],
                [512, 384],
                [384, 256],
                [256, 128],
                [128, 64],
                [512, 384, 256, 128],
                [384, 256, 128, 64],
                [256, 192, 128, 64],
                [512, 256, 128, 64],
                [512, 512, 512, 512],
                [384, 384, 384, 384],
                [512, 384, 192, 96],
                [384, 256, 192, 128],
                [256, 224, 192, 160],
                [512, 512],
                [384, 384],
                [256, 256],
                [192, 192],
            ],
        ),
        "weight_decay": hp.choice("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        # Scheduler Parameters
        "factor": hp.uniform("factor", 0.05, 0.6),
        "patience": hp.choice("patience", [3, 5, 7, 10, 12]),
        # Activation Function Selection
        "activation": hp.choice(
            "activation", ["relu", "leaky_relu", "elu", "swish", "sigmoid", "tanh"]
        ),
        # Batch Normalization Selection
        "use_batch_norm": hp.choice("use_batch_norm", [False, True]),
        # Optimizer Selection and Related Parameter Selection
        "optimizer": hp.choice(
            "optimizer",
            [
                {
                    "type": "adam",
                    "beta1": hp.uniform(
                        "adam_beta1", 0.85, 0.999
                    ),  # Typically close to 1
                    "beta2": hp.uniform(
                        "adam_beta2", 0.9, 0.9999
                    ),  # Typically close to 1
                    "eps": hp.loguniform(
                        "adam_eps", -10, -6
                    ),  # A very small number to prevent any division by zero in the implementation
                },
                {
                    "type": "rmsprop",
                    "alpha": hp.uniform(
                        "rmsprop_alpha", 0.9, 0.9999
                    ),  # Moving average of squared gradient
                    "eps": hp.loguniform(
                        "rmsprop_eps", -10, -6
                    ),  # A small stabilizing term
                },
                {
                    "type": "sgd",
                    "momentum": hp.uniform("sgd_momentum", 0.5, 0.99),  # Momentum term
                    "nesterov": hp.choice(
                        "sgd_nesterov", [True, False]
                    ),  # Whether to use Nesterov acceleration
                },
                {
                    "type": "nadam",
                    "beta1": hp.uniform(
                        "nadam_beta1", 0.85, 0.999
                    ),  # Typically close to 1
                    "beta2": hp.uniform(
                        "nadam_beta2", 0.9, 0.9999
                    ),  # Typically close to 1
                    "eps": hp.loguniform(
                        "nadam_eps", -10, -6
                    ),  # A very small number to prevent any division by zero in the implementation
                },
                {
                    "type": "radam",
                    "beta1": hp.uniform(
                        "radam_beta1", 0.85, 0.999
                    ),  # Typically close to 1
                    "beta2": hp.uniform(
                        "radam_beta2", 0.9, 0.9999
                    ),  # Typically close to 1
                    "eps": hp.loguniform(
                        "radam_eps", -10, -6
                    ),  # A very small number to prevent any division by zero
                },
            ],
        ),
    }

    # Ensure the trials directory exists
    if not os.path.exists("./trials"):
        os.makedirs("./trials")

    trials_save_file = "./trials/nn_trials.pkl"
    # If the file exists, set trials to None so that fmin uses the trials_save_file
    if os.path.exists(trials_save_file):
        trials = None
        print("Saved Trials Object Exists! Picking Up Where we Left Off!")
    else:
        trials = Trials()

    objective = partial(
        nn_objective,
        X_train=X,
        y_train=y,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        device=device,
    )
    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        rstate=np.random.default_rng(105),
        max_evals=500,
        trials=trials,
        trials_save_file=trials_save_file,
    )

    with open(trials_save_file, "rb") as f:
        trials = pickle.load(f)
        print("Loading Trials Object from the Final Saved File!")

    best_params = space_eval(space, trials.argmin)
    save_hyperparameters("nn", best_params)

    return best_params


def get_optimizer(params, optimization_config):
    learning_rate = optimization_config["learning_rate"]
    weight_decay = optimization_config["weight_decay"]

    optimizer_type = optimization_config["optimizer"]["type"]

    if optimizer_type == "adam":
        return torch.optim.Adam(
            params,
            lr=learning_rate,
            betas=(
                optimization_config["optimizer"]["beta1"],
                optimization_config["optimizer"]["beta2"],
            ),
            eps=optimization_config["optimizer"]["eps"],
            weight_decay=weight_decay,
        )

    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=learning_rate,
            alpha=optimization_config["optimizer"]["alpha"],
            eps=optimization_config["optimizer"]["eps"],
            weight_decay=weight_decay,
        )

    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=optimization_config["optimizer"]["momentum"],
            nesterov=optimization_config["optimizer"]["nesterov"],
            weight_decay=weight_decay,
        )

    elif optimizer_type == "nadam":
        return torch.optim.NAdam(
            params,
            lr=learning_rate,
            betas=(
                optimization_config["optimizer"]["beta1"],
                optimization_config["optimizer"]["beta2"],
            ),
            eps=optimization_config["optimizer"]["eps"],
            weight_decay=weight_decay,
        )

    elif optimizer_type == "radam":
        return torch.optim.RAdam(
            params,
            lr=learning_rate,
            betas=(
                optimization_config["optimizer"]["beta1"],
                optimization_config["optimizer"]["beta2"],
            ),
            eps=optimization_config["optimizer"]["eps"],
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(f"Optimizer {optimizer_type} not recognized.")


def compute_accuracy(predictions, labels):
    # Convert sigmoid outputs to binary predictions
    predicted_labels = (predictions > 0.5).float()
    correct = (predicted_labels == labels).float().sum()
    accuracy = correct / len(labels)
    return accuracy.item()


def save_results_to_file(params, accuracy, filename):
    with open(filename, "a") as file:
        file.write("Parameters:\n")
        for key, value in params.items():
            file.write(f"{key}: {value}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write("-" * 40 + "\n")
