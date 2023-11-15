import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def get_adversarial_val_set(train_df, test_df, val_size=0.2):
    """
    Generate a training and validation set where the validation set distribution
    closely matches the test set distribution using adversarial validation, while
    retaining the 'Satisfaction_Rating' column in the final datasets.

    Parameters:
    - train_df: DataFrame containing the training data.
    - test_df: DataFrame containing the test data.
    - val_size: Proportion of the training data to be used as the validation set.

    Returns:
    - training_set: DataFrame for training, including 'Satisfaction_Rating'.
    - validation_set: DataFrame for validation, including 'Satisfaction_Rating'.
    """
    # Save the Satisfaction_Rating column and drop it temporarily from train_df
    satisfaction_ratings = train_df["Satisfaction_Rating"]
    temp_train = train_df.drop("Satisfaction_Rating", axis=1)
    temp_test = test_df.copy()

    # Marking the train and test set
    temp_train["is_test"] = 0
    temp_test["is_test"] = 1

    # Combining train and test sets
    combined_df = pd.concat([temp_train, temp_test], axis=0)

    # Separating features and target for adversarial validation
    X = combined_df.drop("is_test", axis=1)
    y = combined_df["is_test"]

    # Splitting combined data into training and testing for the adversarial model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the LightGBM model
    clf = lgb.LGBMClassifier()
    clf.fit(X_train, y_train)

    # Evaluating the model
    y_pred = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"Adversarial Validation ROC-AUC: {roc_auc}")

    # Predicting the probability of belonging to the test set
    temp_train["test_set_probability"] = clf.predict_proba(temp_train.drop(["is_test"], axis=1))[:, 1]

    # Sorting the training set by the probability and reattaching the Satisfaction_Rating
    temp_train["Satisfaction_Rating"] = satisfaction_ratings
    train_df_sorted = temp_train.sort_values("test_set_probability", ascending=False)

    # Splitting into validation and training sets
    validation_set = train_df_sorted.head(int(len(temp_train) * val_size))
    training_set = train_df_sorted.drop(validation_set.index)

    # Cleaning up the dataframes
    validation_set = validation_set.drop(["is_test", "test_set_probability"], axis=1)
    training_set = training_set.drop(["is_test", "test_set_probability"], axis=1)

    return training_set, validation_set
