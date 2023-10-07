import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the training dataset
train_df = pd.read_csv("../input/train.csv")

# Data Summary
print("Data Summary:")
print(train_df.info())
print("\nSummary Statistics for Numerical Features:")
print(train_df.describe())

# Class Distribution
print("\nClass Distribution for 'Satisfaction Rating':")
print(train_df["Satisfaction Rating"].value_counts())

# Binning for Age
bins = [18, 25, 35, 45, 55, 65, 75, 100]
labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "75+"]
train_df["Age Group"] = pd.cut(train_df["Age"], bins=bins, labels=labels, right=False)

# Binning for Departure and Arrival Delays
delay_bins = [0, 15, 60, 180, 360, 720, 1440]
delay_labels = ["<15min", "15-60min", "1-3hrs", "3-6hrs", "6-12hrs", "12-24hrs"]
train_df["Departure Delay Group"] = pd.cut(
    train_df["Departure Delay in Minutes"], bins=delay_bins, labels=delay_labels, right=False
)
train_df["Arrival Delay Group"] = pd.cut(
    train_df["Arrival Delay in Minutes"], bins=delay_bins, labels=delay_labels, right=False
)

# Interaction Features
train_df["Class_SeatComfort"] = train_df["Class"] + "_" + train_df["Seat Comfort"].astype(str)
train_df["TravelType_FlightDistance"] = (
    train_df["Type of Travel"] + "_" + pd.qcut(train_df["Flight Distance"], q=4).astype(str)
)
train_df["OnlineBooking_OnlineBoarding"] = (
    train_df["Ease of Online booking"].astype(str) + "_" + train_df["Online Boarding"].astype(str)
)
train_df["FlightDistance_SeatComfort"] = (
    pd.qcut(train_df["Flight Distance"], q=4).astype(str) + "_" + train_df["Seat Comfort"].astype(str)
)
train_df["InflightWifi_FlightDistance"] = (
    train_df["Inflight Wifi Service"].astype(str) + "_" + pd.qcut(train_df["Flight Distance"], q=4).astype(str)
)

# Visualization for Age Group and Delays
for feature in ["Age Group", "Departure Delay Group", "Arrival Delay Group"]:
    plt.figure()
    sns.countplot(data=train_df, x=feature, hue="Satisfaction Rating")
    plt.title(f"{feature} vs. Satisfaction Rating")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend(title="Satisfaction Rating", labels=["Neutral / Dissatisfied", "Satisfied"])

# Visualization for Interaction Features
interaction_features = [
    "Class_SeatComfort",
    "TravelType_FlightDistance",
    "OnlineBooking_OnlineBoarding",
    "FlightDistance_SeatComfort",
    "InflightWifi_FlightDistance",
]

for feature in interaction_features:
    plt.figure()
    sns.countplot(data=train_df, x=feature, hue="Satisfaction Rating")
    plt.title(f"{feature} vs. Satisfaction Rating")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend(title="Satisfaction Rating", labels=["Neutral / Dissatisfied", "Satisfied"])
    plt.xticks(rotation=90)

plt.show()
