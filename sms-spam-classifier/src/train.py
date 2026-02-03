import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from preprocess import load_data
from pathlib import Path

# Resolve paths relative to this file to make script runnable from any CWD
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "SMSSpamCollection.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "spam_model.pkl"

df = load_data(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model + vectorizer
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump((model, vectorizer), f)

print(f"✅ Model trained and saved to {MODEL_PATH}.")
