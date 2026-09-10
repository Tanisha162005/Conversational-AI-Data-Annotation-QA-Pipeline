import pandas as pd
import joblib

from sklearn.metrics import accuracy_score

# Load model and test data
model = joblib.load("model/intent_classifier.pkl")
test_df = pd.read_csv("model/test.csv")

# Input and actual labels
X_test = test_df["customer_message"]
y_test = test_df["intent"]

# Predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy Evaluation")
print("-------------------------")
print(f"Accuracy: {accuracy:.2%}")