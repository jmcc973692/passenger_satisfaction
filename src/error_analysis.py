from sklearn.metrics import classification_report, confusion_matrix


def error_analysis(X_valid, y_valid, model, threshold=0.5):
    # Predict on validation set
    y_probs = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    # Confusion Matrix
    cm = confusion_matrix(y_valid, y_pred)
    print("Confusion Matrix:\n", cm)

    # Classification report
    print("\nClassification Report:\n", classification_report(y_valid, y_pred))

    # Instances where predictions were wrong
    wrong_preds = X_valid[
        y_valid != y_pred
    ].copy()  # Use .copy() to explicitly tell pandas that this is a new DataFrame
    wrong_indices = y_valid != y_pred  # This creates a boolean array that you can use with .loc
    wrong_preds.loc[wrong_indices, "true_labels"] = y_valid[wrong_indices]
    wrong_preds.loc[wrong_indices, "predicted_labels"] = y_pred[wrong_indices]

    # You can save or display these instances for a closer look
    # print("\nSamples of incorrectly predicted instances:")
    # print(wrong_preds.sample(10))

    # Return wrong predictions if further analysis is needed
    return wrong_preds
