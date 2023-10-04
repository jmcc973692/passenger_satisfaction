from sklearn.ensemble import GradientBoostingClassifier


def train_model(train_x, train_y):
    # Initialize the Gradient Boosting Classifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Fit the model to the training data
    model.fit(train_x, train_y)

    return model


def evaluate_model(model, test_x, test_y):
    # Make predictions on the test data
    y_pred = model.predict(test_x)

    # Evaluate model accuracy
    accuracy = (y_pred == test_y).mean()
    return accuracy
