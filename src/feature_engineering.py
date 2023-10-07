import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def scale_numerical_features(df, numerical_features):
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df


def log_transform_features(df, features):
    for feature in features:
        df[feature] = np.log(df[feature] + 1)
    return df


def log1p_transform_features(df, features):
    for feature in features:
        df[feature] = np.log1p(df[feature])
    return df


def polynomial_features(df, features):
    poly = PolynomialFeatures(degree=3, include_bias=False)
    for feature in features:
        new_feats = poly.fit_transform(df[[feature]])
        colnames = poly.get_feature_names_out([feature])
        df[colnames] = new_feats
    return df


def bin_features(df):
    age_bins = [0, 18, 35, 60, 100]
    distance_bins = [0, 500, 1500, 5000]
    df["Age_bins"] = pd.cut(df["Age"], bins=age_bins, labels=["Child", "Young_Adult", "Adult", "Senior"])
    df["Flight_Distance_bins"] = pd.cut(
        df["Flight_Distance"], bins=distance_bins, labels=["Short-Haul", "Medium-Haul", "Long-Haul"]
    )
    return df


def encode_categorical_features(df, features):
    encoder = OneHotEncoder(drop="first", sparse=False)
    encoded = encoder.fit_transform(df[features])
    df = pd.concat([df, pd.DataFrame(encoded, columns=encoder.get_feature_names_out(features), index=df.index)], axis=1)
    df.drop(features, axis=1, inplace=True)
    return df


def create_interaction_features(df, feature1, feature2):
    df[f"{feature1}_x_{feature2}"] = df[feature1] * df[feature2]
    return df


def create_encode_categorical_interaction_features(df):
    df["Gender_TypeOfTravel"] = df["Gender"].astype(str) + "_" + df["Type_of_Travel"].astype(str)
    df["FlightDistance_Class"] = df["Flight_Distance"].astype(str) + "_" + df["Class"].astype(str)
    df["Wifi_OnlineBoarding"] = df["Inflight_Wifi_Service"].astype(str) + "_" + df["Online_Boarding"].astype(str)
    df["Food_SeatComfort"] = df["Food_and_Drink"].astype(str) + "_" + df["Seat_Comfort"].astype(str)
    df["OnBoard_InflightService"] = df["On-Board_Service"].astype(str) + "_" + df["Inflight_Service"].astype(str)

    df["Gender_TypeOfTravel"] = df["Gender_TypeOfTravel"].astype("category").cat.codes
    df["FlightDistance_Class"] = df["FlightDistance_Class"].astype("category").cat.codes
    df["Wifi_OnlineBoarding"] = df["Wifi_OnlineBoarding"].astype("category").cat.codes
    df["Food_SeatComfort"] = df["Food_SeatComfort"].astype("category").cat.codes
    df["OnBoard_InflightService"] = df["OnBoard_InflightService"].astype("category").cat.codes

    return df


def create_overall_delay_indicator(df):
    df["Experienced_Delay"] = ((df["Arrival_Delay_in_Minutes"] > 0) | (df["Departure_Delay_in_Minutes"] > 0)).astype(
        int
    )
    return df


def check_for_nans(df, stage):
    nans = df.isna().sum().sum()
    print(f"{stage} NaN count: {nans}")


def perform_feature_engineering(df):
    # Bin Features First
    df = bin_features(df)
    # check_for_nans(df, "After binning")

    # Convert the categorical bins to integer codes
    df["Age_bins"] = df["Age_bins"].cat.codes
    df["Flight_Distance_bins"] = df["Flight_Distance_bins"].cat.codes

    # Add overall delay indicator feature
    df = create_overall_delay_indicator(df)

    # Scaling
    df = scale_numerical_features(df, ["Age"])
    # check_for_nans(df, "After Scaling")

    # Log Transform
    df = log1p_transform_features(df, ["Flight_Distance", "Departure_Delay_in_Minutes", "Arrival_Delay_in_Minutes"])
    # check_for_nans(df, "After log1p transform")

    # Polynomial
    df = polynomial_features(df, ["Flight_Distance", "Age", "Departure_Delay_in_Minutes", "Arrival_Delay_in_Minutes"])
    # check_for_nans(df, "After polynomial")

    # Categorical Interaction Features
    df = create_encode_categorical_interaction_features(df)
    # check_for_nans(df, "After Categorical Interaction Features")

    # One-hot encoding
    df = encode_categorical_features(df, ["Gender", "Customer_Type", "Type_of_Travel", "Class"])

    # Interaction Features
    df = create_interaction_features(df, "Age", "Flight_Distance")
    df = create_interaction_features(df, "Online_Boarding", "Ease_of_Online_booking")
    df = create_interaction_features(df, "Seat_Comfort", "Leg_Room")
    df = create_interaction_features(df, "Inflight_Entertainment", "On-Board_Service")
    df = create_interaction_features(df, "Departure_Delay_in_Minutes", "Arrival_Delay_in_Minutes")
    # check_for_nans(df, "After Interaction Features")

    return df
