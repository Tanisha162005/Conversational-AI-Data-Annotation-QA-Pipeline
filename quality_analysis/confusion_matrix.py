import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load labeled data
DATA_PATH = "annotated_data/labeled_conversations.csv"
df = pd.read_csv(DATA_PATH)

# Compare Annotator 1 with the final adjudicated label
y_true = df["adjudicated_intent"]
y_pred = df["annotator_1_label"]

# Get all intent names
labels = sorted(df["adjudicated_intent"].unique())

# Create confusion matrix
cm = confusion_matrix(
    y_true,
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

plt.title("Annotation Confusion Matrix")
plt.tight_layout()

plt.show()