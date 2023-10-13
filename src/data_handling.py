import pandas as pd


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Replace white spaces with underscores in column names
    train_df.columns = [col.replace(" ", "_") for col in train_df.columns]
    test_df.columns = [col.replace(" ", "_") for col in test_df.columns]

    return train_df, test_df


def handle_missing_data(df):
    # Fill NA in Arrival Delay Data with the Departure Delay Data
    df["Arrival_Delay_in_Minuts"] = df["Arrival_Delay_in_Minutes"].fillna(df["Departure_Delay_in_Minutes"])

    # # Fill NA with the mode of the data
    # for column in df.columns:
    #     mode_val = df[column].mode()[0]
    #     df[column].fillna(mode_val, inplace=True)
    return df


def encode_satisfaction_rating(df):
    df["Satisfaction_Rating"].replace("Neutral / Dissatisfied", 0, inplace=True)
    df["Satisfaction_Rating"].replace("Satisfied", 1, inplace=True)


def drop_low_importance_features(df):
    drop_features = [
        "Departure_Delay_in_Minutes",
        "Departure_Delay_in_Minutes_x_Arrival_Delay_in_Minutes",
        "Flight_Distance",
        "Food_and_Drink",
        "FlightDistance_Class",
        "On-Board_Service",
        "Gender_TypeOfTravel",
        "Gender",
        "Food_SeatComfort",
        "Arrival_Delay_in_Minutes",
        "Age_x_Flight_Distance",
        "Flight_Distance_bins",
        "Age_bins",
        "OnBoard_InflightService",
        "Seat_Comfort_x_Leg_Room",
        "Online_Boarding_x_Ease_of_Online_booking",
    ]
    columns_to_drop = [col for col in drop_features if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    return df


def keep_best_features_only(df, best_features):
    """
    Drops all columns that are not in the best_features list, while ensuring 'Satisfaction_Rating' and 'id' are retained.

    Parameters:
    - df: The input dataframe.
    - best_features: List of best feature column names to keep.

    Returns:
    - Modified dataframe containing only the best features plus 'Satisfaction_Rating' and 'id'.
    """

    # Always retain these columns
    always_keep = ["Satisfaction_Rating", "id"]

    # Only drop columns if they are not in best_features and not in always_keep
    columns_to_drop = [col for col in df.columns if col not in best_features and col not in always_keep]

    df.drop(columns=columns_to_drop, inplace=True)

    return df


def handle_non_responses(train_df, test_df):
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


# def handle_non_responses(train_df, test_df):
#     """
#     Replace non-responses in the survey columns based on strategies provided.

#     Parameters:
#     - train_df (pd.DataFrame): Training data
#     - test_df (pd.DataFrame): Test data

#     Returns:
#     - train_df, test_df with non-responses handled
#     """
#     strategies = {
#         "Ease_of_Online_booking": "mode",
#         "Convenience_of_Departure/Arrival_Time_": "mode",
#         "Baggage_Handling": "mode",
#         "Check-In_Service": "mode",
#         "Gate_Location": "mode",
#         "Online_Boarding": "mode",
#         "Inflight_Wifi_Service": "mode",
#         "Food_and_Drink": "mode",
#         "Seat_Comfort": "mode",
#         "Inflight_Entertainment": "mode",
#         "On-Board_Service": "mode",
#         "Leg_Room": "mode",
#         "Inflight_Service": "mode",
#         "Cleanliness": "mode",
#     }

#     for column, strategy in strategies.items():
#         if strategy == "mean":
#             replace_value = train_df[column].mean()
#         elif strategy == "median":
#             replace_value = train_df[column].median()
#         elif strategy == "mode":
#             replace_value = train_df[column].mode()[0]
#         else:
#             raise ValueError(f"Unknown strategy: {strategy}")

#         train_df[column] = train_df[column].replace(0, replace_value)
#         test_df[column] = test_df[column].replace(0, replace_value)

#     return train_df, test_df


def prepare_data(train_path, test_path):
    train_df, test_df = load_data(train_path, test_path)

    # Handle missing data
    handle_missing_data(train_df)
    handle_missing_data(test_df)

    # Handle the non-responses in the survey data based on some strategy
    handle_non_responses(train_df=train_df, test_df=test_df)

    # Encode the Satisfaction Rating
    encode_satisfaction_rating(train_df)

    return train_df, test_df
