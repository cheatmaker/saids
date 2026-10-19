import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

A = np.array([12, 14, 13, 15, 16])
B = np.array([18, 19, 17, 20, 21])
C = np.array([14, 15, 16, 13, 15])

alpha = 0.05

print("Group A:", A)
print("Group B:", B)
print("Group C:", C)

# One-Way ANOVA
anova = stats.f_oneway(A, B, C)
print("\nANOVA p-value:", anova.pvalue)

if anova.pvalue < alpha:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")

# Kruskal-Wallis Test
kw = stats.kruskal(A, B, C)
print("\nKruskal-Wallis p-value:", kw.pvalue)

if kw.pvalue < alpha:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")

# Display group means
print("\nMean A:", np.mean(A))
print("Mean B:", np.mean(B))
print("Mean C:", np.mean(C))

# Boxplot
plt.boxplot([A, B, C], tick_labels=["Group A", "Group B", "Group C"])
plt.title("Comparison of Multiple Groups")
plt.ylabel("Values")
plt.show()