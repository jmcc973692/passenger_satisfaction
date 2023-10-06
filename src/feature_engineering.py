from sklearn.preprocessing import StandardScaler


def scale_numerical_features(df, numerical_features):
    # Scale specified numerical features using StandardScaler
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df


def create_interaction_features(df, feature1, feature2):
    # Create interaction feature by multiplying two numerical features
    df[f"{feature1}_x_{feature2}"] = df[feature1] * df[feature2]
    return df


def create_encode_categorical_interaction_features(df):
    # Creating categorical interaction features
    df["Gender_TypeOfTravel"] = df["Gender"].astype(str) + "_" + df["Type_of_Travel"].astype(str)
    df["FlightDistance_Class"] = df["Flight_Distance"].astype(str) + "_" + df["Class"].astype(str)
    df["Wifi_OnlineBoarding"] = df["Inflight_Wifi_Service"].astype(str) + "_" + df["Online_Boarding"].astype(str)
    df["Food_SeatComfort"] = df["Food_and_Drink"].astype(str) + "_" + df["Seat_Comfort"].astype(str)
    df["OnBoard_InflightService"] = df["On-Board_Service"].astype(str) + "_" + df["Inflight_Service"].astype(str)

    # Encoding the created categorical interaction features
    df["Gender_TypeOfTravel"] = df["Gender_TypeOfTravel"].astype("category").cat.codes
    df["FlightDistance_Class"] = df["FlightDistance_Class"].astype("category").cat.codes
    df["Wifi_OnlineBoarding"] = df["Wifi_OnlineBoarding"].astype("category").cat.codes
    df["Food_SeatComfort"] = df["Food_SeatComfort"].astype("category").cat.codes
    df["OnBoard_InflightService"] = df["OnBoard_InflightService"].astype("category").cat.codes

    return df


def perform_feature_engineering(df):
    numerical_features_to_scale = ["Age", "Flight_Distance", "Departure_Delay_in_Minutes", "Arrival_Delay_in_Minutes"]
    df = scale_numerical_features(df, numerical_features_to_scale)
    df = create_interaction_features(df, "Age", "Flight_Distance")
    df = create_encode_categorical_interaction_features(df)
    return df
