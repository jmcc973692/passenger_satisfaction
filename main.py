import os
from datetime import datetime

from sklearn.model_selection import cross_val_score

from src.data_handling import prepare_data
from src.feature_engineering import create_interaction_features, scale_numerical_features
from src.model_training import evaluate_model, train_model
from src.submission import load_sample_submission, save_submission, update_submission_structure


def main(train_path, test_path, sample_submission_path, submission_dir):
    train_df, test_df = prepare_data(train_path, test_path)

    # Feature Engineering Steps
    numerical_features_to_scale = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    train_df = scale_numerical_features(train_df, numerical_features_to_scale)
    test_df = scale_numerical_features(test_df, numerical_features_to_scale)

    train_df = create_interaction_features(train_df, "Age", "Flight Distance")
    test_df = create_interaction_features(test_df, "Age", "Flight Distance")

    train_x = train_df.drop(columns=["Satisfaction Rating"])
    train_y = train_df["Satisfaction Rating"]

    model = train_model(train_x, train_y)

    # Check feature importances
    importances = model.feature_importances_
    for feature, importance in zip(train_x.columns, importances):
        print(f"{feature}: {importance:.4f}")

    # Cross-Validation
    scores = cross_val_score(model, train_x, train_y, cv=5)
    print("Cross-validated scores:", scores)
    print("Average score:", scores.mean())

    accuracy = evaluate_model(model, train_x, train_y)
    print(f"Accuracy on training data: {accuracy:.4f}")

    sample_submission_df = load_sample_submission(sample_submission_path)

    test_x = test_df
    y_pred = model.predict(test_x)

    timestamp = datetime.now().strftime("%m_%d_%Y_%H-%M")
    submission_file_name = f"submission_{timestamp}.csv"
    submission_path = os.path.join(submission_dir, submission_file_name)

    update_submission_structure(sample_submission_df, y_pred)
    save_submission(sample_submission_df, submission_path)


if __name__ == "__main__":
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    sample_submission_path = "./input/sample_submission.csv"
    submission_dir = "./submission"

    main(train_path, test_path, sample_submission_path, submission_dir)
