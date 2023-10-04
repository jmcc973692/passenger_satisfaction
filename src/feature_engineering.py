import pandas as pd
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


# Define other feature engineering functions as needed
