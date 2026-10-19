import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

group_A = np.array([21,22,23,24,25,24,26,25,23,22,24,25,26,27,24])
group_B = np.array([27,28,29,30,31,29,30,32,28,29,31,30,29,32,30])

alpha = 0.05

print("Group A:", group_A)
print("Group B:", group_B)

# Normality Test
print("\nShapiro-Wilk Test:")
print("Group A p-value:", stats.shapiro(group_A).pvalue)
print("Group B p-value:", stats.shapiro(group_B).pvalue)

# Equal Variance Test
test = stats.levene(group_A, group_B)
print("\nLevene's Test p-value:", test.pvalue)

# Independent Samples t-Test
test = stats.ttest_ind(group_A, group_B)
print("\nT-Test p-value:", test.pvalue)

if test.pvalue < alpha:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")

# Mean and Standard Deviation
print("\nMean of Group A:", np.mean(group_A))
print("Mean of Group B:", np.mean(group_B))
print("Standard Deviation A:", np.std(group_A, ddof=1))
print("Standard Deviation B:", np.std(group_B, ddof=1))

# Boxplot
plt.boxplot([group_A, group_B], labels=["Group A", "Group B"])
plt.title("Comparison of Group A and Group B")
plt.ylabel("Measured Value")
plt.show()