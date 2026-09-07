import pandas as pd

Data_path = "raw_data/conversations.csv"

df = pd.read_csv(Data_path)

print("Dataset Shape:")
print(df.shape)

print("\n Column Names:")
print(df.columns.tolist())

print("\nfirst in raw:")
print(df.head())

print("\n missing value:")
print(df.isnull().sum())

print("duplicate  rows:")
print(df.duplicated().sum())

print("\nChannel Distribution:")
print(df["channel"].value_counts())


print("\nLanguage Distribution:")
print(df["language"].value_counts())

df["message_length"] = df["customer_message"].str.len()

print("\nMessage Length Statistics:")
print(df["message_length"].describe())