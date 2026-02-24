import pickle
from pathlib import Path
import sys

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "spam_model.pkl"

if not MODEL_PATH.exists():
    sys.exit(f"ERROR: Model file not found at {MODEL_PATH}. Run `python src/train.py` to train the model.")

try:
    with open(MODEL_PATH, "rb") as f:
        model, vectorizer = pickle.load(f)
except ModuleNotFoundError as e:
    sys.exit(
        "ERROR: Failed to load the model because a required package is missing: "
        f"{e}.\nInstall project dependencies with `pip install -r requirements.txt` "
        "using the project's Python environment (for example, `D:/AI-Trainee-Prep/sms-spam-classifier/.venv/Scripts/python.exe -m pip install -r requirements.txt`)."
    )
except Exception as e:
    sys.exit(f"ERROR: Failed to load model: {e}")

def predict_sms(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return "SPAM 🚨" if result == 1 else "HAM(Not Spam) ✅"

def main():
    try:
        while True:
            msg = input("Enter SMS: ")
            print(predict_sms(msg))
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return

if __name__ == "__main__":
    main()

