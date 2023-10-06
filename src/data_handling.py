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

    # Encode The Grouped Categories
    # Assuming the dataframe's respective columns have been transformed to categorical types previously
    df["Age_Group"] = df["Age_Group"].cat.codes
    df["Departure_Delay_Group"] = df["Departure_Delay_Group"].cat.codes
    df["Arrival_Delay_Group"] = df["Arrival_Delay_Group"].cat.codes


def encode_satisfaction_rating(df):
    df["Satisfaction_Rating"].replace("Neutral / Dissatisfied", 0, inplace=True)
    df["Satisfaction_Rating"].replace("Satisfied", 1, inplace=True)


def drop_low_importance_features(df):
    drop_features = [
        "Age",
        "Departure_Delay_in_Minutes",
        "Arrival_Delay_in_Minutes",
    ]
    columns_to_drop = [col for col in drop_features if col in df.columns]
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

    # Binning Age
    bins_age = [0, 18, 35, 50, 65, 100]
    labels_age = ["0-18", "19-35", "36-50", "51-65", "66-100"]
    train_df["Age_Group"] = pd.cut(train_df["Age"], bins=bins_age, labels=labels_age, include_lowest=True)
    test_df["Age_Group"] = pd.cut(test_df["Age"], bins=bins_age, labels=labels_age, include_lowest=True)

    # Binning Delays
    bins_delays = [-1, 15, 60, 180, 360, float("inf")]
    labels_delays = ["<15min", "15-60min", "1-3hrs", "3-6hrs", ">6hrs"]
    train_df["Departure_Delay_Group"] = pd.cut(
        train_df["Departure_Delay_in_Minutes"], bins=bins_delays, labels=labels_delays, include_lowest=True
    )
    train_df["Arrival_Delay_Group"] = pd.cut(
        train_df["Arrival_Delay_in_Minutes"], bins=bins_delays, labels=labels_delays, include_lowest=True
    )

    test_df["Departure_Delay_Group"] = pd.cut(
        test_df["Departure_Delay_in_Minutes"], bins=bins_delays, labels=labels_delays, include_lowest=True
    )
    test_df["Arrival_Delay_Group"] = pd.cut(
        test_df["Arrival_Delay_in_Minutes"], bins=bins_delays, labels=labels_delays, include_lowest=True
    )

    # Encode the training data
    encode_categorical_data(train_df)
    encode_satisfaction_rating(train_df)

    # Encode the testing data
    encode_categorical_data(test_df)

    return train_df, test_df
