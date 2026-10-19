import numpy as np
import matplotlib.pyplot as plt

data = np.array([65, 70, 72, 68, 75, 80, 74, 69, 77, 71])

print("Sample Mean:", np.mean(data))

# Bootstrap Resampling
means = []
for i in range(5000):
    sample = np.random.choice(data, size=len(data), replace=True)
    means.append(np.mean(sample))

means = np.array(means)

# Results
print("Bootstrap Standard Error:", np.std(means, ddof=1))
print("95% Confidence Interval:", np.percentile(means, [2.5, 97.5]))

# Plot
plt.hist(means, bins=30, edgecolor="black")
plt.axvline(np.mean(data), linestyle="--", label="Sample Mean")
plt.xlabel("Bootstrap Means")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of Sample Mean")
plt.legend()
plt.show()