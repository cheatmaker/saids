import pandas as pd

df = pd.read_csv("data.csv")

print("First 5 rows:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# Handle missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Data transformation
df = pd.get_dummies(df, drop_first=True)

print("\nCleaned Data:\n", df.head())
print("\nDataset Information:")
df.info()