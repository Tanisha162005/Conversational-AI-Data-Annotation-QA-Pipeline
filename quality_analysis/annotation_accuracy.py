import pandas as pd

# Load annotated dataset
DATA_PATH = "annotated_data/labeled_conversations.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# Check annotation status
print("\nReview Status:")
print(df["review_status"].value_counts())

# Compare each annotator with the adjudicated label
for annotator in ["annotator_1_label", "annotator_2_label", "annotator_3_label"]:
    
    accuracy = (
        df[annotator] == df["adjudicated_intent"]
    ).mean()

    print(f"\n{annotator} agreement with adjudicated label:")
    print(f"{accuracy:.2%}")