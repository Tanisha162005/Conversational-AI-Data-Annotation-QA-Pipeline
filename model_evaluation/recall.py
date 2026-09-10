import pandas as pd
import joblib

from sklearn.metrics import recall_score

# Load model and test data
model = joblib.load("model/intent_classifier.pkl")
test_df = pd.read_csv("model/test.csv")

# Input and actual labels
X_test = test_df["customer_message"]
y_test = test_df["intent"]

# Make predictions
y_pred = model.predict(X_test)

# Calculate recall
recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("Model Recall Evaluation")
print("-----------------------")
print(f"Macro Recall: {recall:.2%}")