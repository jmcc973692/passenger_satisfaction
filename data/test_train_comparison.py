import warnings

# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew

# Load the datasets
train_df = pd.read_csv("./output/train_full_feature_set.csv")
test_df = pd.read_csv("./output/test_full_feature_set.csv")

# Drop the target variable from train_df
train_df = train_df.drop(columns="Satisfaction_Rating")

# Since all features are either numerical or one-hot encoded, we'll treat everything as numerical
numerical_features = train_df.columns

# Distribution Analysis
for feature in numerical_features:
    plt.figure(figsize=(12, 6))
    sns.kdeplot(train_df[feature], label="Train", shade=True)
    sns.kdeplot(test_df[feature], label="Test", shade=True)
    plt.title(f"Distribution of {feature}")
    plt.legend()
    safe_feature_name = feature.replace("/", "_")
    plt.savefig(f"./output/plots/{safe_feature_name}_distribution.png")

# Feature Statistics
stats_df = pd.DataFrame(
    index=numerical_features,
    columns=["Train Mean", "Test Mean", "Train Std", "Test Std", "Train Skew", "Test Skew", "Train Kurt", "Test Kurt"],
)
for feature in numerical_features:
    stats_df.at[feature, "Train Mean"] = train_df[feature].mean()
    stats_df.at[feature, "Test Mean"] = test_df[feature].mean()
    stats_df.at[feature, "Train Std"] = train_df[feature].std()
    stats_df.at[feature, "Test Std"] = test_df[feature].std()
    stats_df.at[feature, "Train Skew"] = skew(train_df[feature])
    stats_df.at[feature, "Test Skew"] = skew(test_df[feature])
    stats_df.at[feature, "Train Kurt"] = kurtosis(train_df[feature])
    stats_df.at[feature, "Test Kurt"] = kurtosis(test_df[feature])

stats_df.to_csv("./output/feature_statistics.csv")

# Stability Index
stability_df = pd.DataFrame(index=numerical_features, columns=["Stability Index"])
for feature in numerical_features:
    train_counts = train_df[feature].value_counts(normalize=True)
    test_counts = test_df[feature].value_counts(normalize=True)
    stability_index = (train_counts - test_counts).pow(2).sum() / len(train_df)
    stability_df.at[feature, "Stability Index"] = stability_index

stability_df.to_csv("./output/stability_indices.csv")

# Correlation Analysis
correlation_matrix = train_df.corr()
plt.figure(figsize=(18, 18))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.savefig("./output/correlation_matrix.png")
