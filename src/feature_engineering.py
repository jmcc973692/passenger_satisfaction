import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def scale_numerical_features(df, numerical_features):
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df


def log1p_transform_features(df, features):
    for feature in features:
        df[feature] = np.log1p(df[feature])
    return df


def onehot_encode_features(df, features_to_encode):
    """
    One-hot encode the specified columns of a DataFrame using sklearn's OneHotEncoder.

    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - features_to_encode (list): List of columns to be one-hot encoded.

    Returns:
    - pd.DataFrame: DataFrame with the specified columns one-hot encoded.
    """
    encoder = OneHotEncoder(drop=None, sparse=False)

    for feature in features_to_encode:
        # Fit and transform the feature
        encoded_data = encoder.fit_transform(df[[feature]])

        # Determine column names
        if len(df[feature].unique()) == 2:
            col_names = [f"{feature}_{encoder.categories_[0][1]}"]
            encoded_data = encoded_data[:, 1:]  # Keep only the second column
        else:
            col_names = encoder.get_feature_names_out([feature])

        # Convert to DataFrame and rename columns as necessary
        encoded_df = pd.DataFrame(encoded_data, columns=col_names).astype(int)
        encoded_df.columns = encoded_df.columns.str.replace(" ", "_")

        # Drop the original feature from the dataframe and join the encoded DataFrame
        df = df.drop(feature, axis=1).join(encoded_df)

    return df


def polynomial_features(df, features):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    for feature in features:
        new_feats = poly.fit_transform(df[[feature]])
        colnames = poly.get_feature_names_out([feature])
        df[colnames] = new_feats
    return df


def bin_features(df):
    age_bins = [0, 18, 35, 60, 100]
    distance_bins = [0, 500, 1500, 5000]
    delay_difference_bins = [-np.inf, -5, 5, np.inf]
    df["Age_bins"] = pd.cut(
        df["Age"], bins=age_bins, labels=["Child", "Young_Adult", "Adult", "Senior"]
    )
    df["Flight_Distance_bins"] = pd.cut(
        df["Flight_Distance"],
        bins=distance_bins,
        labels=["Short-Haul", "Medium-Haul", "Long-Haul"],
    )
    df["Flight_Delay_Difference"] = pd.cut(
        df["Flight_Delay_Difference_Minutes"],
        bins=delay_difference_bins,
        labels=["Made_Up_Time", "No_Delay_Change", "Lost_Time"],
    )
    return df


def encode_categorical_features(df, features):
    for feature in features:
        df[feature] = df[feature].astype("category").cat.codes + 1
    return df


def create_multiplicative_interaction_feature(df, feature1, feature2):
    df[f"{feature1}_x_{feature2}"] = df[feature1] * df[feature2]
    return df


def create_overall_delay_indicator(df):
    df["Experienced_Delay"] = (
        (df["Arrival_Delay_in_Minutes"] > 0) | (df["Departure_Delay_in_Minutes"] > 0)
    ).astype(int)
    return df


def create_aggregate_scores(df):
    # Overall Service Quality Score
    service_features = [
        "Ease_of_Online_booking",
        "Check-In_Service",
        "Online_Boarding",
        "Inflight_Wifi_Service",
        "On-Board_Service",
        "Inflight_Service",
    ]
    df["Service_Quality_Score"] = df[service_features].sum(axis=1)

    # Flight Comfort Score
    comfort_features = [
        "Seat_Comfort",
        "Leg_Room",
        "Inflight_Entertainment",
        "Food_and_Drink",
        "Cleanliness",
    ]
    df["Comfort_Score"] = df[comfort_features].sum(axis=1)

    # Convenience Score
    convenience_features = [
        "Convenience_of_Departure/Arrival_Time_",
        "Baggage_Handling",
        "Gate_Location",
    ]
    df["Convenience_Score"] = df[convenience_features].sum(axis=1)

    return df


def frequency_encoding(df, columns):
    """
    Perform frequency encoding on specified columns.

    Parameters:
    - df (pd.DataFrame): Training/Test Data
    - columns (list): List of columns to perform frequency encoding on

    Returns:
    - df with frequency encoded columns
    """

    for column in columns:
        # Create a frequency map based on the training data
        freq_map = df[column].value_counts(normalize=True)

        # Map the frequencies onto the dataframes
        df[column + "_freq"] = df[column].map(freq_map)

    return df


def one_hot_encode_survey_features(df):
    """
    One-hot-encodes specified columns with values 1-5, dropping the 0 feature.

    Args:
    - df (pd.DataFrame): DataFrame with the columns to be encoded.

    Returns:
    - pd.DataFrame: One-hot-encoded DataFrame.
    """

    columns = [
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

    for col in columns:
        # Ensure the column has categories 1-5, even if they are not present
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep="_", dtype=int)
        for i in range(1, 6):
            if f"{col}_{i}" not in dummies.columns:
                dummies[f"{col}_{i}"] = 0

        # Drop the 0 feature to avoid collinearity
        dummies.drop(columns=[f"{col}_0"], errors="ignore", inplace=True)

        # Replace the original column with the encoded columns
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df


def feedback_consistency_check_feature(df):
    """
    Add features based on the difference in related feedback columns.

    Parameters:
    - df (pd.DataFrame): Input data

    Returns:
    - df with added feedback consistency check features
    """

    # Create difference features
    df["diff_inflight_onboard_service"] = (
        df["Inflight_Service"] - df["On-Board_Service"]
    )
    df["diff_seatcomfort_legroom"] = df["Seat_Comfort"] - df["Leg_Room"]
    df["diff_wifi_onlineboarding"] = df["Inflight_Wifi_Service"] - df["Online_Boarding"]
    df["diff_food_cleanliness"] = df["Food_and_Drink"] - df["Cleanliness"]

    return df


def check_for_nans(df, stage):
    nans = df.isna().sum().sum()
    print(f"{stage} NaN count: {nans}")


def perform_feature_engineering(df):
    # Add delay indicator features
    df = create_overall_delay_indicator(df)
    df["Total_Delay_Minutes"] = (
        df["Departure_Delay_in_Minutes"] + df["Arrival_Delay_in_Minutes"]
    )
    df["Flight_Delay_Difference_Minutes"] = (
        df["Arrival_Delay_in_Minutes"] - df["Departure_Delay_in_Minutes"]
    )

    # Bin Features
    df = bin_features(df)
    # check_for_nans(df, "After binning")

    # # Scaling
    # df = scale_numerical_features(df, ["Age"])
    # # check_for_nans(df, "After Scaling")

    # # Log Transform
    # df = log1p_transform_features(
    #     df, ["Flight_Distance", "Departure_Delay_in_Minutes", "Arrival_Delay_in_Minutes", "Total_Delay_Minutes"]
    # )
    # # check_for_nans(df, "After log1p transform")

    # Polynomial
    df = polynomial_features(
        df,
        [
            "Flight_Distance",
            "Age",
            "Departure_Delay_in_Minutes",
            "Arrival_Delay_in_Minutes",
        ],
    )
    # check_for_nans(df, "After polynomial")

    # Create Frequency Encoding Features
    columns_to_encode = ["Type_of_Travel", "Class", "Gender", "Customer_Type"]
    df = frequency_encoding(df, columns_to_encode)
    # check_for_nans(df, "After Frequency Encoding")

    # One Hot Encode Categorical Features
    df = onehot_encode_features(
        df,
        [
            "Gender",
            "Customer_Type",
            "Type_of_Travel",
            "Class",
            "Age_bins",
            "Flight_Distance_bins",
            "Flight_Delay_Difference",
        ],
    )
    # check_for_nans(df, "After One Hot Encoding")

    # Categorical Multiplicative Interaction Features
    df = create_multiplicative_interaction_feature(
        df, "Online_Boarding", "Ease_of_Online_booking"
    )
    df = create_multiplicative_interaction_feature(df, "Seat_Comfort", "Leg_Room")
    df = create_multiplicative_interaction_feature(
        df, "Inflight_Entertainment", "Flight_Distance"
    )
    df = create_multiplicative_interaction_feature(
        df, "Food_and_Drink", "Flight_Distance"
    )
    df = create_multiplicative_interaction_feature(df, "Age", "Type_of_Travel_Personal")
    df = create_multiplicative_interaction_feature(
        df, "Gender_Male", "Inflight_Wifi_Service"
    )
    df = create_multiplicative_interaction_feature(
        df, "Departure_Delay_in_Minutes", "Arrival_Delay_in_Minutes"
    )
    df = create_multiplicative_interaction_feature(df, "Age", "Inflight_Entertainment")
    df = create_multiplicative_interaction_feature(df, "Class_Economy", "Cleanliness")
    df = create_multiplicative_interaction_feature(
        df, "Class_Economy_Plus", "Cleanliness"
    )
    df = create_multiplicative_interaction_feature(
        df, "Customer_Type_Non-Loyal_Customer", "On-Board_Service"
    )
    df = create_multiplicative_interaction_feature(
        df, "Class_Economy", "Flight_Distance"
    )
    df = create_multiplicative_interaction_feature(
        df, "Class_Economy_Plus", "Flight_Distance"
    )
    # check_for_nans(df, "After Interaction Features")

    # Aggregate Features
    df = create_aggregate_scores(df)
    # check_for_nans(df, "After Aggregate Features")

    # Feedback Consistency Features
    df = feedback_consistency_check_feature(df)
    # check_for_nans(df, "After Feedback Consistency Features")

    return df


def perform_feature_engineering_lgbm(df):
    # Separate the Data into Data Types
    # numerical_columns = ["Age", "Flight_Distance", "Departure_Delay_in_Minutes", "Arrival_Delay_in_Minutes"]
    categorical_columns = ["Gender", "Customer_Type", "Type_of_Travel", "Class"]

    df = polynomial_features(df, ["Age"])
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)
    # for feature in categorical_columns:
    #     df[feature] = df[feature].astype("category")
    df = one_hot_encode_survey_features(df)
    df.columns = df.columns.str.replace(" ", "_")

    # df = scale_numerical_features(df, numerical_columns)

    return df


def perform_feature_engineering_nn(df):
    # Separate the Data into Data Types
    numerical_columns = [
        "Age",
        "Flight_Distance",
        "Departure_Delay_in_Minutes",
        "Arrival_Delay_in_Minutes",
    ]
    categorical_columns = ["Gender", "Customer_Type", "Type_of_Travel", "Class"]

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)
    df = one_hot_encode_survey_features(df)
    df.columns = df.columns.str.replace(" ", "_")

    df = scale_numerical_features(df, numerical_columns)

    return df
