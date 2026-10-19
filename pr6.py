import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 6])

# Correlation
r, p = pearsonr(x.flatten(), y)
print("Correlation Coefficient:", r)
print("P-value:", p)

# Linear Regression
model = LinearRegression()
model.fit(x, y)

print("\nSlope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Prediction
y_pred = model.predict(x)
print("R-squared:", r2_score(y, y_pred))

# Plot
plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, color="red", label="Regression Line")
plt.title("Correlation and Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()