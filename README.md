# 🧠 Emotion AI Chatbot

A real-time emotion detection chatbot that understands how you feel and responds with empathy. Built with Flask, scikit-learn, and NLTK — deployed on Railway.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) ![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange?logo=scikit-learn) ![Railway](https://img.shields.io/badge/Deployed-Railway-blueviolet?logo=railway)

---

Live demo: https://web-production-04543.up.railway.app/

## ✨ Features

- **Real-time emotion detection** — classifies your message into 7 emotions: Happy, Love, Sad, Stressed, Angry, Surprised, and Neutral
- **Dual detection system** — combines a TF-IDF + Logistic Regression ML model with a synonym lookup dictionary for higher accuracy
- **Empathetic responses** — tailored replies based on the detected emotion
- **Emotion timeline chart** — visualizes how your mood shifts across the conversation
- **Confidence scores** — shows top 2 predicted emotions with percentage confidence
- **Session stats** — tracks message count and dominant mood of the session
- **Responsive UI** — clean chat interface that works on desktop and mobile

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Gunicorn |
| ML Model | scikit-learn (Logistic Regression + TF-IDF) |
| NLP | NLTK (stopwords, WordNet lemmatizer) |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Dataset | Tweet Emotions dataset (40,000 tweets, 13 emotion labels) |
| Deployment | Railway |

---

## 🤖 How It Works

1. User sends a message
2. The backend preprocesses the text (lowercasing, URL removal, stop word filtering)
3. A **synonym lookup** scans for emotion keywords first (fast path)
4. If no keyword match, the **ML model** (TF-IDF vectorizer + Logistic Regression) predicts the emotion
5. The higher-confidence result is chosen and a matching empathetic reply is returned
6. The frontend updates the chat, emotion badge, timeline chart, and session stats

---

## 🚀 Running Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Emotional-Chatbot.git
cd Emotional-Chatbot

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open `http://localhost:5000` in your browser.

> **Note:** On first run, the app will automatically train the ML model from `tweet_emotions.csv` and save the pkl files. This takes about 30–60 seconds.

---

## 📁 Project Structure

```
├── app.py                  # Flask app, ML model, prediction logic
├── tweet_emotions.csv      # Training dataset
├── requirements.txt        # Python dependencies
├── railway.json            # Railway deployment config
├── Procfile                # Gunicorn start command
├── runtime.txt             # Python version
├── templates/
│   └── index.html          # Chat UI
└── static/
    ├── script.js           # Frontend logic, chart, API calls
    └── style.css           # Styling
```

---

## 🎭 Supported Emotions

| Emotion | Examples |
|---|---|
| 😊 Happy | "I'm feeling great today", "I'm so excited" |
| 💖 Love | "I love spending time with family", "I miss her" |
| 💙 Sad | "I feel so lonely", "I'm heartbroken" |
| 😟 Stressed | "I'm worried about my exam", "I can't handle this" |
| 😠 Angry | "I'm so frustrated", "This makes me furious" |
| 😲 Surprised | "I can't believe that happened", "That was unexpected" |
| 🙂 Neutral | General statements with no strong emotional tone |

---

## ☁️ Deployment

This app is deployed on **Railway** with automatic GitHub deploys.

On first deploy, the app trains the ML model in a background thread so the server starts instantly. Once training completes (~60 seconds), all predictions are live.


