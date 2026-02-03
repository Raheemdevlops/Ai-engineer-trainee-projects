📩 SMS Spam Classifier

This project builds a machine learning model to classify SMS messages as Spam or Ham using the UCI SMS Spam Collection dataset.

📊 Dataset

Source: UCI Machine Learning Repository
Total messages: 5,572
Labels: Spam, Ham
⚙️ Approach

Text preprocessing
TF-IDF vectorization
Multinomial Naive Bayes classifier
📈 Results

Accuracy: ~97%
High precision and recall for spam detection
🛠 Tech Stack
Python
scikit-learn
pandas

⏩ How to Run
pip install -r requirements.txt
python src/train.py
python src/predict.py

