import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report

# Paths
MODEL_PATH = "model/intent_classifier.pkl"
VALIDATION_PATH = "model/validation.csv"

# Load trained model
model = joblib.load(MODEL_PATH)

# Load validation data
df = pd.read_csv(VALIDATION_PATH)

# Input and actual labels
X_validation = df["customer_message"]
y_validation = df["intent"]

# Make predictions
y_pred = model.predict(X_validation)

# Calculate accuracy
accuracy = accuracy_score(y_validation, y_pred)

print("Validation Results")
print("------------------")
print(f"Validation Accuracy: {accuracy:.2%}")

# Detailed results for every intent
print("\nClassification Report:")
print(
    classification_report(
        y_validation,
        y_pred,
        zero_division=0
    )
)