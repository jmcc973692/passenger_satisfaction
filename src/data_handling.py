import pandas as pd


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def handle_missing_data(df):
    df.dropna(subset=["Arrival Delay in Minutes"], inplace=True)


def encode_categorical_data(df):
    # Encode Gender
    df["Gender"].replace("Female", 0, inplace=True)
    df["Gender"].replace("Male", 1, inplace=True)

    # Encode Customer Type
    df["Customer Type"].replace("Loyal Customer", 0, inplace=True)
    df["Customer Type"].replace("Non-Loyal Customer", 1, inplace=True)

    # Encode Type of Travel
    df["Type of Travel"].replace("Business", 0, inplace=True)
    df["Type of Travel"].replace("Personal", 1, inplace=True)

    # Encode Class
    df["Class"].replace("Business", 0, inplace=True)
    df["Class"].replace("Economy", 1, inplace=True)
    df["Class"].replace("Economy Plus", 2, inplace=True)


def encode_satisfaction_rating(df):
    df["Satisfaction Rating"].replace("Neutral / Dissatisfied", 0, inplace=True)
    df["Satisfaction Rating"].replace("Satisfied", 1, inplace=True)


def prepare_data(train_path, test_path):
    train_df, test_df = load_data(train_path, test_path)

    # Prepare the training data
    handle_missing_data(train_df)
    encode_categorical_data(train_df)
    encode_satisfaction_rating(train_df)

    # Prepare the testing data
    handle_missing_data(test_df)
    encode_categorical_data(test_df)

    return train_df, test_df
