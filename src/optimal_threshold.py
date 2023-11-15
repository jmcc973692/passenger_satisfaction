from functools import partial, update_wrapper

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, cross_val_score


def custom_accuracy_score(y_true, y_pred_probs, threshold):
    y_pred = (y_pred_probs >= threshold).astype(int)
    return accuracy_score(y_true, y_pred)


def threshold_scorer(threshold):
    # Define a custom scorer that uses the threshold
    def score_func(y_true, y_pred_probs):
        return custom_accuracy_score(y_true, y_pred_probs, threshold)

    # Set the __name__ attribute to avoid the AttributeError
    score_func.__name__ = f"custom_accuracy_score_{threshold}"
    return make_scorer(score_func, needs_proba=True)


def threshold_objective(threshold, X, y, model):
    # Get the custom scorer with the current threshold
    scorer = threshold_scorer(threshold)

    # Calculate cross-validation
    # score using the scorer
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y, cv=rkf, scoring=scorer, n_jobs=-1)

    mean_cv_score = scores.mean()
    return {"loss": -mean_cv_score, "status": STATUS_OK}


def find_optimal_threshold(X, y, model):
    space = hp.normal("threshold", mu=0.5, sigma=0.1)

    trials = Trials()
    objective = partial(threshold_objective, X=X, y=y, model=model)
    optimal_threshold = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

    print(f"Optimal Threshold = {optimal_threshold['threshold']}")
    return optimal_threshold["threshold"]
