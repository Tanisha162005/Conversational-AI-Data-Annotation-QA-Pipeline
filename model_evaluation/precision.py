import pandas as pd
import joblib

from sklearn.metrics import precision_score

# Load model and test data
model = joblib.load("model/intent_classifier.pkl")
test_df = pd.read_csv("model/test.csv")

# Input and actual labels
X_test = test_df["customer_message"]
y_test = test_df["intent"]

# Make predictions
y_pred = model.predict(X_test)

# Calculate precision
precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("Model Precision Evaluation")
print("--------------------------")
print(f"Macro Precision: {precision:.2%}")