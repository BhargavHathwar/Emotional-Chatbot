from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "emotion_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))


# --------------------------------
# Home Page
# --------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# --------------------------------
# Emotion Prediction
# --------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    message = data["message"]

    text = message.lower()

    # convert text to vector
    vector = vectorizer.transform([text])

    # machine learning prediction
    emotion = model.predict(vector)[0]


    # --------------------------------
    # Keyword override rules
    # --------------------------------

    if any(word in text for word in ["angry","mad","furious","rage"]):
        emotion = "angry"

    elif any(word in text for word in ["happy","excited","joy","great","awesome"]):
        emotion = "happy"

    elif any(word in text for word in ["sad","depressed","upset","unhappy"]):
        emotion = "sad"

    elif any(word in text for word in ["stress","worried","anxious","nervous"]):
        emotion = "stress"

    elif any(word in text for word in ["calm","okay","fine","normal"]):
        emotion = "neutral"


    # --------------------------------
    # Chatbot responses
    # --------------------------------

    responses = {

        "happy": "I'm really happy to hear that! 😊 Keep smiling and enjoying the moment.",

        "sad": "I'm sorry you're feeling sad. 💙 Remember that tough moments pass.",

        "stress": "It sounds like you're feeling stressed. 😟 Try taking a deep breath and relax.",

        "angry": "I understand you're feeling angry. 😠 Maybe step away for a moment and breathe.",

        "neutral": "I see. Tell me more about how you're feeling. 🙂"
    }

    reply = responses.get(emotion, "I understand how you feel. 🙂")

    return jsonify({
        "emotion": emotion,
        "reply": reply
    })


# --------------------------------
# Run Server
# --------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)