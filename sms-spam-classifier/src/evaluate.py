import pickle
from sklearn.metrics import classification_report
from preprocess import load_data

df = load_data("../data/SMSSpamCollection")

with open("../model/spam_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

X_vec = vectorizer.transform(df["message"])
preds = model.predict(X_vec)

print(classification_report(df["label"], preds))
