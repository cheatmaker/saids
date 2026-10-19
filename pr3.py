import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Random variable
data = np.random.randint(1, 7, 1000)

print("First 5 values:", data[:5])
print("Mean:", np.mean(data))
print("Variance:", np.var(data))

# Probability distribution
sns.histplot(data, discrete=True, stat="probability")
plt.title("Probability Distribution")
plt.xlabel("Random Variable")
plt.ylabel("Probability")
plt.show()

# Normal distribution
x = np.random.normal(0, 1, 1000)
sns.histplot(x, kde=True)
plt.title("Normal Distribution")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()