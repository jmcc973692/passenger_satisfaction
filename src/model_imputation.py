import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def handle_non_responses_mode(train_df, test_df):
    """
    Replace non-responses in the survey columns based on strategies provided.

    Parameters:
    - train_df (pd.DataFrame): Training data
    - test_df (pd.DataFrame): Test data

    Returns:
    - train_df, test_df with non-responses handled
    """
    strategies = {
        "Ease_of_Online_booking": "mode",
        "Convenience_of_Departure/Arrival_Time_": "mode",
        "Baggage_Handling": "mode",
        "Check-In_Service": "mode",
        "Gate_Location": "mode",
        "Online_Boarding": "mode",
        "Inflight_Wifi_Service": "mode",
        "Food_and_Drink": "mode",
        "Seat_Comfort": "mode",
        "Inflight_Entertainment": "mode",
        "On-Board_Service": "mode",
        "Leg_Room": "mode",
        "Inflight_Service": "mode",
        "Cleanliness": "mode",
    }

    for column, strategy in strategies.items():
        if strategy == "mean":
            replace_value = train_df[column].mean()
        elif strategy == "median":
            replace_value = train_df[column].median()
        elif strategy == "mode":
            replace_value = train_df[column].mode()[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        train_df[column] = train_df[column].replace(0, replace_value)
        test_df[column] = test_df[column].replace(0, replace_value)

    return train_df, test_df


def handle_non_responses_group(train_df, test_df):
    """
    Replace non-responses in the survey columns based on strategies provided.

    Parameters:
    - train_df (pd.DataFrame): Training data
    - test_df (pd.DataFrame): Test data

    Returns:
    - train_df, test_df with non-responses handled
    """

    # Group the features
    service_features = [
        "Ease_of_Online_booking",
        "Check-In_Service",
        "Online_Boarding",
        "Inflight_Wifi_Service",
        "On-Board_Service",
        "Inflight_Service",
    ]
    comfort_features = ["Seat_Comfort", "Leg_Room", "Inflight_Entertainment", "Food_and_Drink", "Cleanliness"]
    convenience_features = ["Convenience_of_Departure/Arrival_Time_", "Baggage_Handling", "Gate_Location"]

    # Combine into a dictionary for easier processing
    grouped_features = {"service": service_features, "comfort": comfort_features, "convenience": convenience_features}

    # Iterate over both train and test dataframes
    for df in [train_df, test_df]:
        for group, features in grouped_features.items():
            for feature in features:
                mask = df[feature] == 0
                other_features = [f for f in features if f != feature]

                # Compute the mean of other features in the group for rows where feature value is 0
                replacement = df.loc[mask, other_features].mean(axis=1)

                # If all the features in that group for that row are 0, replace with the mode
                replacement.fillna(df[feature].mode()[0], inplace=True)

                df.loc[mask, feature] = replacement

    return train_df, test_df


def handle_non_responses_knn(train_df, test_df, n_neighbors=5):
    """
    Replace non-responses in the survey columns using KNN imputation.

    Parameters:
    - train_df (pd.DataFrame): Training data
    - test_df (pd.DataFrame): Test data
    - n_neighbors (int): Number of neighbors for KNN imputation

    Returns:
    - train_df, test_df with non-responses handled
    """

    # Features to be considered for imputation
    survey_features = [
        "Ease_of_Online_booking",
        "Check-In_Service",
        "Online_Boarding",
        "Inflight_Wifi_Service",
        "On-Board_Service",
        "Inflight_Service",
        "Seat_Comfort",
        "Leg_Room",
        "Inflight_Entertainment",
        "Food_and_Drink",
        "Cleanliness",
        "Convenience_of_Departure/Arrival_Time_",
        "Baggage_Handling",
        "Gate_Location",
    ]

    # Replace 0 with NaN in survey columns for both training and test data
    for df in [train_df, test_df]:
        df[survey_features] = df[survey_features].replace(0, np.nan)

    # Use KNN imputation to replace NaNs
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Impute training data
    train_imputed = imputer.fit_transform(train_df)
    train_df = pd.DataFrame(train_imputed, columns=train_df.columns)

    # Impute test data
    test_imputed = imputer.transform(test_df)
    test_df = pd.DataFrame(test_imputed, columns=test_df.columns)

    return train_df, test_df


def handle_0_responses(train_df, test_df, strategy="knn"):
    if strategy == "knn":
        train_df, test_df = handle_non_responses_knn(train_df, test_df, n_neighbors=5)
    elif strategy == "group":
        train_df, test_df = handle_non_responses_group(train_df, test_df)
    elif strategy == "mode":
        train_df, test_df = handle_non_responses_mode(train_df, test_df)
    return train_df, test_df
