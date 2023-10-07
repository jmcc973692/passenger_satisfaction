import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
train_path = "../input/train.csv"
test_path = "../input/test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Define the columns of interest
numerical_columns = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]

# Transformations
# Scaling age
scaler = StandardScaler()
train_df["Age"] = scaler.fit_transform(train_df[["Age"]])
test_df["Age"] = scaler.transform(test_df[["Age"]])

# Log transform for Departure Delay and Arrival Delay (Adding 1 to avoid log(0))
train_df["Departure Delay in Minutes"] = np.log1p(train_df["Departure Delay in Minutes"])
train_df["Arrival Delay in Minutes"] = np.log1p(train_df["Arrival Delay in Minutes"])
test_df["Departure Delay in Minutes"] = np.log1p(test_df["Departure Delay in Minutes"])
test_df["Arrival Delay in Minutes"] = np.log1p(test_df["Arrival Delay in Minutes"])

# Square root and log transform for Flight Distance
train_df["Flight Distance_sqrt"] = np.sqrt(train_df["Flight Distance"])
train_df["Flight Distance_log"] = np.log1p(train_df["Flight Distance"])
test_df["Flight Distance_sqrt"] = np.sqrt(test_df["Flight Distance"])
test_df["Flight Distance_log"] = np.log1p(test_df["Flight Distance"])

# Adding the transformed Flight Distance columns to the numerical columns list
numerical_columns.extend(["Flight Distance_sqrt", "Flight Distance_log"])

# Analyze the numerical data columns
analysis = pd.DataFrame(index=numerical_columns)

analysis["Mean"] = train_df[numerical_columns].mean()
analysis["Median"] = train_df[numerical_columns].median()
analysis["Std Dev"] = train_df[numerical_columns].std()
analysis["Min"] = train_df[numerical_columns].min()
analysis["Max"] = train_df[numerical_columns].max()
analysis["25th Percentile"] = train_df[numerical_columns].quantile(0.25)
analysis["75th Percentile"] = train_df[numerical_columns].quantile(0.75)
analysis["IQR"] = analysis["75th Percentile"] - analysis["25th Percentile"]
analysis["Skewness"] = train_df[numerical_columns].skew()
analysis["Kurtosis"] = train_df[numerical_columns].kurtosis()

# Save the analysis to the output directory
output_path = "../output/numerical_data_analysis_transformed.csv"
analysis.to_csv(output_path)

print(analysis)
