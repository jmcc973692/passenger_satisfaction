import pandas as pd


def load_sample_submission(sample_submission_path):
    return pd.read_csv(sample_submission_path)


def update_submission_structure(submission_df, y_pred):
    # Replace the "Satisfaction Rating" column in submission_df with predictions
    submission_df["Satisfaction Rating"] = y_pred
    # Replace numerical values with corresponding labels
    submission_df["Satisfaction Rating"].replace(0, "Neutral / Dissatisfied", inplace=True)
    submission_df["Satisfaction Rating"].replace(1, "Satisfied", inplace=True)


def save_submission(submission_df, submission_path):
    # Save the updated submission DataFrame to a CSV file
    submission_df.to_csv(submission_path, index=False)
