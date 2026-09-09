import pandas as pd
from sklearn.model_selection import train_test_split

# Load prepared ML dataset
INPUT_PATH = "model/model_dataset.csv"

df = pd.read_csv(INPUT_PATH)

print("Total dataset:")
print(df.shape)

# First split:
# 80% training, 20% temporary data
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["intent"]
)

# Second split:
# Divide the temporary 20% equally
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["intent"]
)

# Save the three datasets
train_df.to_csv(
    "model/train.csv",
    index=False
)

validation_df.to_csv(
    "model/validation.csv",
    index=False
)

test_df.to_csv(
    "model/test.csv",
    index=False
)

# Display results
print("\nTraining dataset:")
print(train_df.shape)

print("\nValidation dataset:")
print(validation_df.shape)

print("\nTest dataset:")
print(test_df.shape)