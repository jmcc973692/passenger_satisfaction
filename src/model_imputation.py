import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


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

    # Step 1: Identify columns that have 0's and need imputation
    survey_columns = [
        "Ease_of_Online_booking",
        "Convenience_of_Departure/Arrival_Time_",
        "Baggage_Handling",
        "Check-In_Service",
        "Gate_Location",
        "Online_Boarding",
        "Inflight_Wifi_Service",
        "Food_and_Drink",
        "Seat_Comfort",
        "Inflight_Entertainment",
        "On-Board_Service",
        "Leg_Room",
        "Inflight_Service",
        "Cleanliness",
    ]

    # Step 2: Temporarily preprocess data for KNN imputation
    temp_train = train_df.copy()
    temp_test = test_df.copy()

    # Label encoding categorical columns for KNN
    categorical_columns = ["Gender", "Customer_Type", "Type_of_Travel", "Class"]
    for col in categorical_columns:
        le = LabelEncoder()
        temp_train[col] = le.fit_transform(temp_train[col])
        temp_test[col] = le.transform(temp_test[col])

    # Step 3: Apply KNN imputation on selected columns
    imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=0)
    temp_train[survey_columns] = imputer.fit_transform(temp_train[survey_columns])
    temp_test[survey_columns] = imputer.transform(temp_test[survey_columns])

    # Step 4: Replace only the survey columns in original dataframes
    for col in survey_columns:
        train_df[col] = temp_train[col]
        test_df[col] = temp_test[col]

    return train_df, test_df


def handle_0_responses(train_df, test_df, strategy="knn"):
    if strategy == "knn":
        train_df, test_df = handle_non_responses_knn(train_df, test_df, n_neighbors=5)
    elif strategy == "group":
        train_df, test_df = handle_non_responses_group(train_df, test_df)
    elif strategy == "mode":
        train_df, test_df = handle_non_responses_mode(train_df, test_df)
    else:
        pass
    return train_df, test_df
