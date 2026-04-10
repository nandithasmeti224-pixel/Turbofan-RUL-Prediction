 Crop Production Prediction Project
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 2. Load Dataset
# 👉 Replace with your actual file name
df = pd.read_csv(""C:\Users\sures\OneDrive\Desktop\Crop_data.csv"")

print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 3. Data Cleaning
# Fill missing values
df.fillna(method='ffill', inplace=True)

# 4. Convert Categorical Columns
le = LabelEncoder()
categorical_cols = ['Crop', 'Variety', 'state', 'Season', 'Unit', 'Recommended Zone'
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
print("\nAfter Encoding:")
print(df.head())

# 5. Define Features and Target
X = df.drop("production", axis=1)
y = df["production"]

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train Model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# 10. Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.title("Actual vs Predicted Production")
plt.show()

# 11. Predict New Sample
# 👉 Modify values according to your dataset format
sample = [1, 2, 3, 100, 1, 0, 5000, 2]
prediction = model.predict([sample])
print("\nPredicted Production for Sample:", prediction)
