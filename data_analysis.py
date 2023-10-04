import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the training dataset
train_df = pd.read_csv("./input/train.csv")

# Data Summary
print("Data Summary:")
print(train_df.info())
print("\nSummary Statistics for Numerical Features:")
print(train_df.describe())

# Class Distribution
print("\nClass Distribution for 'Satisfaction Rating':")
print(train_df["Satisfaction Rating"].value_counts())

# Univariate Analysis for Numerical Features
numerical_features = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]

for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

# Univariate Analysis for Categorical Features
categorical_features = ["Gender", "Customer Type", "Type of Travel", "Class"]

for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=train_df, x=feature, hue="Satisfaction Rating")
    plt.title(f"{feature} vs. Satisfaction Rating")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend(title="Satisfaction Rating", labels=["Neutral / Dissatisfied", "Satisfied"])
    plt.show()

# Bivariate Analysis for Numerical Features vs. Target
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=train_df, x="Satisfaction Rating", y=feature)
    plt.title(f"{feature} vs. Satisfaction Rating")
    plt.xlabel("Satisfaction Rating")
    plt.ylabel(feature)
    plt.show()

# Missing Data Analysis
missing_data = train_df.isnull().sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_data.index, y=missing_data.values)
plt.title("Missing Data Analysis")
plt.xlabel("Features")
plt.ylabel("Missing Values Count")
plt.xticks(rotation=90)
plt.show()
