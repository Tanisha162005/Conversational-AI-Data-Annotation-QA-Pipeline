import pandas as pd

# Input and output paths
INPUT_PATH = "annotated_data/labeled_conversations.csv"
OUTPUT_PATH = "model/model_dataset.csv"

# Load labeled data
df = pd.read_csv(INPUT_PATH)

print("Original dataset:")
print(df.shape)

# Keep only the fields needed for ML
model_df = df[
    ["conversation_id", "customer_message", "adjudicated_intent"]
].copy()

# Rename the final label to "intent"
model_df = model_df.rename(
    columns={
        "adjudicated_intent": "intent"
    }
)

# Remove rows with missing ML fields
model_df = model_df.dropna(
    subset=["customer_message", "intent"]
)

# Remove duplicate conversations
model_df = model_df.drop_duplicates(
    subset=["conversation_id"]
)

# Save ML dataset
model_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nML dataset:")
print(model_df.shape)

print("\nColumns:")
print(model_df.columns.tolist())

print("\nIntent distribution:")
print(model_df["intent"].value_counts())

print("\nML dataset saved to:")
print(OUTPUT_PATH)