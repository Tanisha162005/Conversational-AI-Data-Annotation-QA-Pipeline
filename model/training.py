import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load training data
TRAIN_PATH = "model/train.csv"

train_df = pd.read_csv(TRAIN_PATH)

print("Training dataset:")
print(train_df.shape)

# Separate input and target
X_train = train_df["customer_message"]
y_train = train_df["intent"]

# Create ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

# Train the model
print("\nTraining model...")

model.fit(X_train, y_train)

# Save trained model
MODEL_PATH = "model/intent_classifier.pkl"

joblib.dump(model, MODEL_PATH)

print("Model saved to:")
print(MODEL_PATH)

print("Model training completed!")