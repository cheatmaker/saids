import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")

print("First 5 rows:\n", df.head())
print("\nDataset Information:")
df.info()
print("\nStatistical Summary:\n", df.describe())
print("\nRows and Columns:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Histogram
df.hist(figsize=(10, 6))
plt.show()

# Correlation Heatmap
sns.heatmap(df.select_dtypes("number").corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Boxplot
df.select_dtypes("number").plot(kind="box")
plt.title("Boxplot")
plt.show()