import pandas as pd


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Replace white spaces with underscores in column names
    train_df.columns = [col.replace(" ", "_") for col in train_df.columns]
    test_df.columns = [col.replace(" ", "_") for col in test_df.columns]

    return train_df, test_df


def handle_missing_data(df):
    for column in df.columns:
        mode_val = df[column].mode()[0]
        df[column].fillna(mode_val, inplace=True)
    return df


def encode_categorical_data(df):
    # Encode Gender
    df["Gender"].replace("Female", 0, inplace=True)
    df["Gender"].replace("Male", 1, inplace=True)

    # Encode Customer Type
    df["Customer_Type"].replace("Loyal Customer", 0, inplace=True)
    df["Customer_Type"].replace("Non-Loyal Customer", 1, inplace=True)

    # Encode Type of Travel
    df["Type_of_Travel"].replace("Business", 0, inplace=True)
    df["Type_of_Travel"].replace("Personal", 1, inplace=True)

    # Encode Class
    df["Class"].replace("Business", 0, inplace=True)
    df["Class"].replace("Economy", 1, inplace=True)
    df["Class"].replace("Economy Plus", 2, inplace=True)


def encode_satisfaction_rating(df):
    df["Satisfaction_Rating"].replace("Neutral / Dissatisfied", 0, inplace=True)
    df["Satisfaction_Rating"].replace("Satisfied", 1, inplace=True)


def drop_low_importance_features(df):
    drop_features = [
        "Departure_Delay_in_Minutes",
        "Flight_Distance",
        "FlightDistance_Class",
        "Food_and_Drink",
        "Age_x_Flight_Distance",
        "Food_SeatComfort",
        "Inflight_Entertainment",  # assuming this exists based on pattern
        "Customer_Type_Loyal Customer",  # inverse of "Customer_Type_Non-Loyal Customer"
        "Gender_Female",  # inverse of "Gender_Male"
        "Type_of_Travel_Business",  # inverse of "Type_of_Travel_Personal"
        "Class_Business",  # assuming this exists based on pattern
        "Online_Support"  # another mock feature
        # ... you can add more features to drop
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
