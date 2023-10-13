from sklearn.metrics import classification_report, confusion_matrix


def error_analysis(X_valid, y_valid, model):
    # Predict on validation set
    y_pred = model.predict(X_valid)

    # Confusion Matrix
    cm = confusion_matrix(y_valid, y_pred)
    print("Confusion Matrix:\n", cm)

    # Classification report
    print("\nClassification Report:\n", classification_report(y_valid, y_pred))

    # Instances where predictions were wrong
    wrong_preds = X_valid[y_valid != y_pred]
    wrong_preds["true_labels"] = y_valid[y_valid != y_pred]
    wrong_preds["predicted_labels"] = y_pred[y_valid != y_pred]

    # You can save or display these instances for a closer look
    print("\nSamples of incorrectly predicted instances:")
    print(wrong_preds.sample(10))

    # Return wrong predictions if further analysis is needed
    return wrong_preds
