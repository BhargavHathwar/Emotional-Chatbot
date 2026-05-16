from flask import Flask, render_template, request, jsonify
import pickle
import os
import random
import logging
import pandas as pd
import re
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------
# Global model state
# --------------------------------
MODEL_LOADED = False
model = None
vectorizer = None

# --------------------------------
# Multiple Responses Per Emotion
# --------------------------------
responses = {
    "happy": [
        "That's wonderful to hear! 😊 Keep riding that positive wave!",
        "Love the good vibes! 🌟 What's making you feel so great today?",
        "You're glowing with happiness! 😄 Enjoy every bit of it!",
    ],
    "sad": [
        "I'm sorry you're feeling sad. 💙 Remember, tough moments always pass.",
        "It's okay to feel down sometimes. 🌧️ I'm here if you want to talk.",
        "Sending you warmth. 💙 You're stronger than you think.",
    ],
    "stress": [
        "Sounds like you're carrying a lot right now. 😟 Take a slow, deep breath — one thing at a time.",
        "Stress can feel overwhelming. 🌿 Try stepping away for 5 minutes and resetting.",
        "You've handled tough moments before — you've got this. 💪 What's weighing on you most?",
    ],
    "angry": [
        "I hear you — that frustration is valid. 😠 Take a moment to breathe before reacting.",
        "Anger is a signal something matters to you. 🔥 What's behind the frustration?",
        "It's okay to feel angry. 😤 Step away for a bit and let it settle.",
    ],
    "neutral": [
        "I see. Tell me more about how you're feeling. 🙂",
        "Sounds like a calm moment. 😌 Anything on your mind?",
        "Got it. I'm here whenever you want to share more. 🙂",
    ],
}

# --------------------------------
# Keyword Override (negation-aware)
# --------------------------------
NEGATIONS = {"not", "no", "never", "don't", "didn't", "isn't", "wasn't", "can't", "won't"}

KEYWORD_MAP = {
    "angry":   ["angry", "mad", "furious", "rage"],
    "happy":   ["happy", "excited", "joy", "great", "awesome"],
    "sad":     ["sad", "depressed", "upset", "unhappy"],
    "stress":  ["stress", "worried", "anxious", "nervous"],
    "neutral": ["calm", "okay", "fine", "normal"],
}

def keyword_override(text, ml_emotion):
    words = text.lower().split()
    for i, word in enumerate(words):
        prev = words[i - 1] if i > 0 else ""
        if prev in NEGATIONS:
            continue
        for emotion, keywords in KEYWORD_MAP.items():
            if word in keywords:
                return emotion
    return ml_emotion

# --------------------------------
# Train and Save Model
# --------------------------------
def train_and_save():
    logger.info("Training model from scratch...")
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        words = text.split()
        words = [w for w in words if w not in stop_words]
        words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    data = pd.read_csv(os.path.join(BASE_DIR, "tweet_emotions.csv"))
    data = data[["content", "sentiment"]]
    data.columns = ["text", "emotion"]
    data["text"] = data["text"].apply(preprocess)

    emotion_map = {
        "happiness": "happy", "love": "happy", "fun": "happy", "enthusiasm": "happy",
        "sadness": "sad", "empty": "sad",
        "worry": "stress",
        "anger": "angry", "hate": "angry",
        "boredom": "neutral", "neutral": "neutral", "relief": "neutral", "surprise": "neutral"
    }
    data["emotion"] = data["emotion"].map(emotion_map)
    data = data.dropna()

    vec = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
    X = vec.fit_transform(data["text"])
    y = data["emotion"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = LogisticRegression(max_iter=2000)
    mdl.fit(X_train, y_train)

    pickle.dump(mdl, open(os.path.join(BASE_DIR, "emotion_model.pkl"), "wb"))
    pickle.dump(vec, open(os.path.join(BASE_DIR, "vectorizer.pkl"), "wb"))
    logger.info("Model trained and saved successfully.")
    return mdl, vec

# --------------------------------
# Load or Retrain in Background
# --------------------------------
def load_or_train():
    global model, vectorizer, MODEL_LOADED
    try:
        model_path = os.path.join(BASE_DIR, "emotion_model.pkl")
        vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
        model = pickle.load(open(model_path, "rb"))
        vectorizer = pickle.load(open(vectorizer_path, "rb"))
        vectorizer.transform(["test"])
        model.predict(vectorizer.transform(["test"]))
        logger.info("Model loaded successfully.")
        MODEL_LOADED = True
    except Exception as e:
        logger.warning(f"Model load failed ({e}), retraining...")
        try:
            model, vectorizer = train_and_save()
            MODEL_LOADED = True
            logger.info("Retraining complete.")
        except Exception as e2:
            logger.error(f"Retraining failed: {e2}")
            MODEL_LOADED = False

# Start in background so gunicorn binds to port immediately
threading.Thread(target=load_or_train, daemon=True).start()

# --------------------------------
# Home Page
# --------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# --------------------------------
# Health Check
# --------------------------------
@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if MODEL_LOADED else "loading",
        "model_loaded": MODEL_LOADED
    })

# --------------------------------
# Emotion Prediction
# --------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model is still loading, please wait 30 seconds and try again."}), 503

    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = str(data["message"]).strip()
    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        text = message.lower()
        vector = vectorizer.transform([text])
        emotion = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]
        classes = model.classes_
        confidence = round(float(max(proba)) * 100)

        top2 = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:2]
        top2_result = [{"emotion": e, "confidence": round(float(c) * 100)} for e, c in top2]

        emotion = keyword_override(text, emotion)
        reply = random.choice(responses.get(emotion, ["I understand how you feel. 🙂"]))

        logger.info(f"Predicted: {emotion} ({confidence}%) | Input: {message[:60]}")

        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "top2": top2_result,
            "reply": reply
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed. Please try again."}), 500

# --------------------------------
# Run Server
# --------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)