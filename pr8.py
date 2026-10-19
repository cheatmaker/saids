import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Create dataset
df = pd.DataFrame({
    "Study_Hours": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5],
    "Result": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
})

X = df[["Study_Hours"]]
y = df["Result"]

# Train model
model = LogisticRegression()
model.fit(X, y)

print("Intercept:", model.intercept_[0])
print("Coefficient:", model.coef_[0][0])
print("Odds Ratio:", np.exp(model.coef_[0][0]))

# Prediction
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

df["Probability"] = prob
df["Predicted Class"] = pred
print("\nPredictions:\n", df)
print("\nAccuracy:", accuracy_score(y, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, pred))

# Predict for a new student
print("\nProbability of passing after 4.2 hours:",
      model.predict_proba(pd.DataFrame({"Study_Hours": [4.2]}))[0][1])

# Plot
x = pd.DataFrame({"Study_Hours": np.linspace(1, 6.5, 100)})
plt.scatter(X, y, label="Actual Data")
plt.plot(x, model.predict_proba(x)[:, 1], label="Logistic Regression")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression")
plt.legend()
plt.grid()
plt.show()