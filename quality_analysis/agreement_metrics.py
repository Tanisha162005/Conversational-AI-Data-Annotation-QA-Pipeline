import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load labeled data
DATA_PATH = "annotated_data/labeled_conversations.csv"
df = pd.read_csv(DATA_PATH)

# Get annotator labels
annotator_1 = df["annotator_1_label"]
annotator_2 = df["annotator_2_label"]
annotator_3 = df["annotator_3_label"]

# Calculate pairwise Cohen's Kappa
kappa_12 = cohen_kappa_score(annotator_1, annotator_2)
kappa_13 = cohen_kappa_score(annotator_1, annotator_3)
kappa_23 = cohen_kappa_score(annotator_2, annotator_3)

print("Inter-Annotator Agreement")
print("--------------------------")

print(f"Annotator 1 vs Annotator 2: {kappa_12:.3f}")
print(f"Annotator 1 vs Annotator 3: {kappa_13:.3f}")
print(f"Annotator 2 vs Annotator 3: {kappa_23:.3f}")

# Average agreement
average_kappa = (kappa_12 + kappa_13 + kappa_23) / 3

print(f"\nAverage Cohen's Kappa: {average_kappa:.3f}")