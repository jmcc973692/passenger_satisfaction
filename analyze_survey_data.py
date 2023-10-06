import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def analyze_column(df, column, file):
    file.write(f"Analysis for {column}:\n")

    # Count of each response
    counts = df[column].value_counts().sort_index()
    file.write("\nResponse Counts:\n")
    file.write(str(counts) + "\n")

    # Basic statistics
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode().iloc[0]
    file.write(f"\nMean: {mean:.2f}\nMedian: {median}\nMode: {mode}\n")
    file.write("-" * 40 + "\n")


def main(data_path, output_path):
    df = load_data(data_path)

    # List of survey columns to analyze
    survey_columns = [
        "Ease of Online booking",
        "Convenience of Departure/Arrival Time ",
        "Baggage Handling",
        "Check-In Service",
        "Gate Location",
        "Online Boarding",
        "Inflight Wifi Service",
        "Food and Drink",
        "Seat Comfort",
        "Inflight Entertainment",
        "On-Board Service",
        "Leg Room",
        "Inflight Service",
        "Cleanliness",
    ]

    with open(output_path, "w") as file:
        for column in survey_columns:
            analyze_column(df, column, file)


if __name__ == "__main__":
    data_path = "./input/train.csv"
    output_path = "./output/survey_analysis.txt"
    main(data_path, output_path)
