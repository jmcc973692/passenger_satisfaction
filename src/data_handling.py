import pandas as pd

from model_imputation import handle_0_responses


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Replace white spaces with underscores in column names
    train_df.columns = [col.replace(" ", "_") for col in train_df.columns]
    test_df.columns = [col.replace(" ", "_") for col in test_df.columns]

    return train_df, test_df


def handle_missing_data(df):
    # Fill NA in Arrival Delay Data with the Departure Delay Data
    df["Arrival_Delay_in_Minuts"] = df["Arrival_Delay_in_Minutes"].fillna(
        df["Departure_Delay_in_Minutes"], inplace=True
    )

    # # Fill NA with the mode of the data
    # for column in df.columns:
    #     mode_val = df[column].mode()[0]
    #     df[column].fillna(mode_val, inplace=True)
    return df


def encode_satisfaction_rating(df):
    df["Satisfaction_Rating"].replace("Neutral / Dissatisfied", 0, inplace=True)
    df["Satisfaction_Rating"].replace("Satisfied", 1, inplace=True)


def prepare_data(train_path, test_path):
    train_df, test_df = load_data(train_path, test_path)

    # Handle missing data
    handle_missing_data(train_df)
    handle_missing_data(test_df)

    # Handle the non-responses in the survey data based on some strategy
    handle_0_responses(train_df=train_df, test_df=test_df, strategy="knn")

    # Encode the Satisfaction Rating
    encode_satisfaction_rating(train_df)

    return train_df, test_df
