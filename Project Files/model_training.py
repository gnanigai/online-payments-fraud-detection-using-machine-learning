import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("payment.csv")

# Drop unnecessary columns
df.drop(["nameOrig", "nameDest", "isFlaggedFraud", "step"], axis=1, inplace=True)

# Encode categorical column
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])

# Input and Output
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("../flask/payments.pkl", "wb"))
pickle.dump(model, open("payments.pkl", "wb"))


print("Model saved successfully as payments.pkl")
