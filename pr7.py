import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset Shape:", X.shape)
print("\nFirst Five Records:\n", X.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Without Scaling
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
acc1 = accuracy_score(y_test, model.predict(X_test))

# Normalization
scaler = MinMaxScaler()
X_train_n = scaler.fit_transform(X_train)
X_test_n = scaler.transform(X_test)

model.fit(X_train_n, y_train)
acc2 = accuracy_score(y_test, model.predict(X_test_n))

# Standardization
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model.fit(X_train_s, y_train)
acc3 = accuracy_score(y_test, model.predict(X_test_s))

# Compare accuracy
results = pd.DataFrame({
    "Scaling Method": ["Without Scaling", "Normalization", "Standardization"],
    "Accuracy": [acc1, acc2, acc3]
})

print("\nModel Performance:\n", results)