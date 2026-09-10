import pandas as pd
import joblib

from sklearn.metrics import f1_score

# Load model and test data
model = joblib.load("model/intent_classifier.pkl")
test_df = pd.read_csv("model/test.csv")

# Input and actual labels
X_test = test_df["customer_message"]
y_test = test_df["intent"]

# Make predictions
y_pred = model.predict(X_test)

# Calculate F1-score
f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("Model F1-Score Evaluation")
print("-------------------------")
print(f"Macro F1-Score: {f1:.2%}")