import pandas as pd
from sklearn.linear_model import LinearRegression

# Load CSV file (assume 4 columns: 3 features + 1 target)
print("Running ML example...")
df = pd.read_csv("data.csv")

# First 3 columns as features (X), last column as target (y)
X = df.iloc[:, :3]
y = df.iloc[:, 3]

# Train model
model = LinearRegression()
model.fit(X, y)

# Print coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Running ML example...")
# Predict on same data (just for demo)
predictions = model.predict(X)
print("Running ML example...")
print("Predictions:", predictions[:10])  # show first 10
print("Running ML example...")
