BEST_FEATURES = [
    "Age_x_Type_of_Travel_Personal",
    "Baggage_Handling",
    "Check-In_Service",
    "Class_Economy",
    "Class_Economy_Plus_x_Cleanliness",
    "Cleanliness",
    "Convenience_of_Departure/Arrival_Time_",
    "Customer_Type_Non-Loyal_Customer_x_On-Board_Service",
    "Departure_Delay_in_Minutes",
    "Ease_of_Online_booking",
    "Flight_Delay_Difference_Made_Up_Time",
    "Gate_Location",
    "Inflight_Entertainment",
    "Inflight_Service",
    "Inflight_Wifi_Service",
    "Leg_Room",
    "On-Board_Service",
    "Online_Boarding",
    "Online_Boarding_x_Ease_of_Online_booking",
    "Seat_Comfort",
    "Seat_Comfort_x_Leg_Room",
    "Type_of_Travel_Personal",
    "Age",
    "Flight_Distance_bins_Long-Haul",
    "Flight_Distance_bins_Medium-Haul",
]


def keep_best_features_only(df, best_features=BEST_FEATURES):
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
