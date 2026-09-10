import pandas as pd
import joblib

# Load model and test data
model = joblib.load("model/intent_classifier.pkl")
test_df = pd.read_csv("model/test.csv")

# Make predictions
test_df["predicted_intent"] = model.predict(
    test_df["customer_message"]
)

# Keep only incorrect predictions
errors = test_df[
    test_df["intent"] != test_df["predicted_intent"]
].copy()

print("Total test conversations:")
print(len(test_df))

print("\nTotal incorrect predictions:")
print(len(errors))

print("\nIncorrect Predictions:")
print(
    errors[
        [
            "conversation_id",
            "customer_message",
            "intent",
            "predicted_intent"
        ]
    ].to_string(index=False)
)

# Save errors for further analysis
errors.to_csv(
    "quality_analysis/model_errors.csv",
    index=False
)

print("\nError file saved to:")
print("quality_analysis/model_errors.csv")