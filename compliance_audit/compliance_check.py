import pandas as pd
import re

# Load annotated dataset
DATA_PATH = "annotated_data/labeled_conversations.csv"

df = pd.read_csv(DATA_PATH)

print("Starting Compliance Audit...")
print(f"Total records: {len(df)}")


# --------------------------------------------------
# 1. Missing values
# --------------------------------------------------

# Check required fields only
required_columns = [
    "conversation_id",
    "channel",
    "customer_message",
    "language",
    "synthetic"
]

missing_values = df[required_columns].isnull().sum().sum()

print("\nMissing values in required fields:")
print(missing_values)


# --------------------------------------------------
# 2. Duplicate conversation IDs
# --------------------------------------------------

duplicate_ids = df["conversation_id"].duplicated().sum()

print("\nDuplicate conversation IDs:")
print(duplicate_ids)


# --------------------------------------------------
# 3. Email detection
# --------------------------------------------------

email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

email_count = df["customer_message"].str.contains(
    email_pattern,
    regex=True,
    na=False
).sum()

print("\nPossible email addresses:")
print(email_count)


# --------------------------------------------------
# 4. Phone number detection
# --------------------------------------------------

phone_pattern = r"\b(?:\+?\d[\d\s\-()]{7,}\d)\b"

phone_count = df["customer_message"].str.contains(
    phone_pattern,
    regex=True,
    na=False
).sum()

print("\nPossible phone numbers:")
print(phone_count)


# --------------------------------------------------
# 5. URL detection
# --------------------------------------------------

url_pattern = r"https?://\S+|www\.\S+"

url_count = df["customer_message"].str.contains(
    url_pattern,
    regex=True,
    na=False
).sum()

print("\nPossible URLs:")
print(url_count)


# --------------------------------------------------
# 6. Card-number-like sequence detection
# --------------------------------------------------

card_pattern = r"\b(?:\d[ -]*?){13,19}\b"

card_count = df["customer_message"].str.contains(
    card_pattern,
    regex=True,
    na=False
).sum()

print("\nPossible card numbers:")
print(card_count)


# --------------------------------------------------
# 7. Synthetic data verification
# --------------------------------------------------

synthetic_count = df["synthetic"].eq(True).sum()

print("\nRecords marked as synthetic:")
print(synthetic_count)


# --------------------------------------------------
# 8. Final compliance result
# --------------------------------------------------

if (
    missing_values == 0
    and duplicate_ids == 0
    and email_count == 0
    and phone_count == 0
    and url_count == 0
    and card_count == 0
    and synthetic_count == len(df)
):
    compliance_status = "PASS"
else:
    compliance_status = "REVIEW REQUIRED"


print("\n==============================")
print("COMPLIANCE AUDIT RESULT")
print("==============================")
print(f"Status: {compliance_status}")