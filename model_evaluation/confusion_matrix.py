import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load model and test data
model = joblib.load("model/intent_classifier.pkl")
test_df = pd.read_csv("model/test.csv")

# Input and actual labels
X_test = test_df["customer_message"]
y_test = test_df["intent"]

# Make predictions
y_pred = model.predict(X_test)

# Get all intents
labels = sorted(y_test.unique())

# Create confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(14, 14))

display.plot(
    ax=ax,
    xticks_rotation=90
)

plt.title("Model Confusion Matrix - Test Set")
plt.tight_layout()

# Save the figure
plt.savefig(
    "model_evaluation/model_confusion_matrix.png",
    dpi=300
)

plt.show()