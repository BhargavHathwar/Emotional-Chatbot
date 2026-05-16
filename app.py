from flask import Flask, render_template, request, jsonify
import pickle
import os
import random
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model_path = os.path.join(BASE_DIR, "emotion_model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    logger.info("Model and vectorizer loaded successfully.")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    MODEL_LOADED = False

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not available"}), 503

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)