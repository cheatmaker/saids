import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_iris()
X, y = data.data, data.target

print("Dataset Shape:", X.shape)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=200)

scores = cross_val_score(model, X, y, cv=kf)

print("\nValidation Scores:", scores)
print("Mean Accuracy:", np.mean(scores))
print("Standard Deviation:", np.std(scores))

# Compare scores
for i, score in enumerate(scores):
    print("Fold", i + 1, "Accuracy:", round(score, 4))