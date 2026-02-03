import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("model/spam_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

@app.route("/")
def home():
    return "SMS Spam Classifier is running 🟢"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("message", "")
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    return jsonify({
        "message": text,
        "prediction": "spam" if pred == 1 else "ham"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
