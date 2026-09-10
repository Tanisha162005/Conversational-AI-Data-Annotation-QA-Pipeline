import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report

# Paths
MODEL_PATH = "model/intent_classifier.pkl"
TEST_PATH = "model/test.csv"

# Load trained model
model = joblib.load(MODEL_PATH)

# Load test data
df = pd.read_csv(TEST_PATH)

print("Test dataset:")
print(df.shape)

# Input and actual labels
X_test = df["customer_message"]
y_test = df["intent"]

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Test Results")
print("------------------")
print(f"Test Accuracy: {accuracy:.2%}")

# Detailed evaluation
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)