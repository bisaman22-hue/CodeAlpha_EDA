import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset load karo
df = pd.read_csv("train.csv")

# First 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Basic statistics
print("\nStatistics:")
print(df.describe())

# Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.show()

# Gender Distribution
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.savefig("gender_distribution.png")
plt.show()

# Passenger Class Distribution
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.savefig("passenger_class_distribution.png")
plt.show()