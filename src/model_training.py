import xgboost as xgb


def xgboost_train_model(train_x, train_y):
    model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=100, max_depth=5)
    model.fit(train_x, train_y)
    return model


def evaluate_model(model, val_x, val_y):
    return model.score(val_x, val_y)
