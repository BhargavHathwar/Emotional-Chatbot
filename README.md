# Emotion AI Chatbot

<div align="center">

## AI-Powered Emotion Detection & Support Chatbot

A modern emotion-aware chatbot built using Machine Learning, Flask, NLP, and an interactive frontend UI. The chatbot analyzes user text in real time, predicts emotions using a trained ML model, and responds with supportive and context-aware replies.

</div>

---

# Features

* Real-time emotion detection from text
* Emotion-aware chatbot responses
* Interactive and modern chat UI
* Emotion timeline visualization using Chart.js
* NLP preprocessing with NLTK
* TF-IDF text vectorization
* Logistic Regression ML model
* Multiple dynamic responses for each emotion
* Negation-aware keyword override system
* Automatic model training and loading
* Flask backend with REST API
* Deployment-ready structure

---

# Supported Emotions

| Emotion | Description                  |
| ------- | ---------------------------- |
| Happy   | Positive and joyful emotions |
| Sad     | Low or negative emotions     |
| Stress  | Anxiety, worry, or pressure  |
| Angry   | Frustration or anger         |
| Neutral | Calm or balanced state       |

---

# Tech Stack

## Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js

## Backend

* Python
* Flask

## Machine Learning & NLP

* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* NLTK
* Pandas
* NumPy

---

# Project Structure

```bash
Emotion-Chatbot/
│
├── app.py                  # Main Flask application
├── train_model.py          # Model training script
├── test_model.py           # Test prediction script
├── requirements.txt        # Dependencies
├── runtime.txt             # Python runtime version
├── Procfile                # Deployment configuration
├── tweet_emotions.csv      # Dataset
│
├── templates/
│   └── index.html          # Frontend UI
│
├── static/
│   ├── style.css           # Styling
│   └── script.js           # Frontend logic
│
└── emotion_model.pkl       # Trained model
└── vectorizer.pkl          # Saved vectorizer
```

---

# How It Works

## 1. Text Preprocessing

The chatbot cleans and processes user text using:

* Lowercasing
* URL removal
* Special character removal
* Stopword removal
* Lemmatization

---

## 2. Feature Extraction

The processed text is converted into numerical vectors using:

```python
TfidfVectorizer()
```

with unigram and bigram support.

---

## 3. Machine Learning Model

The chatbot uses:

```python
LogisticRegression()
```

for emotion classification.

---

## 4. Emotion Prediction

The trained model predicts emotions from user input in real time.

---

## 5. Smart Response System

The chatbot responds using:

* Emotion-specific replies
* Randomized responses
* Negation-aware keyword override logic

Example:

```text
"I am not happy"
```

will not incorrectly classify the message as happy.

---

# Installation & Setup

## Clone the Repository

```bash
git clone https://github.com/your-username/emotion-ai-chatbot.git
cd emotion-ai-chatbot
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train the Model

```bash
python train_model.py
```

This generates:

* `emotion_model.pkl`
* `vectorizer.pkl`

---

# Run the Application

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

---

# Example Chat

```text
User: I feel stressed about my exams.

Bot: Stress can feel overwhelming. Try stepping away for 5 minutes and resetting.
```

---

# Machine Learning Workflow

```text
User Input
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression Model
   ↓
Emotion Prediction
   ↓
Emotion-Based Response
```

---

# Dataset

The chatbot is trained using the `tweet_emotions.csv` dataset.

Original emotions are mapped into 5 simplified categories:

| Original Emotion | Mapped Emotion |
| ---------------- | -------------- |
| happiness        | happy          |
| sadness          | sad            |
| worry            | stress         |
| anger            | angry          |
| neutral          | neutral        |

---

# Deployment

This project is deployment-ready and can be hosted on:

* Render
* Railway
* Heroku
* Replit
* PythonAnywhere

The repository already includes:

* `Procfile`
* `runtime.txt`

for cloud deployment.

---

# Future Improvements

* Voice emotion detection
* Multi-language support
* Deep learning models
* User authentication
* Emotion analytics dashboard
* Chat history storage
* Personalized responses
* Mobile application

---

# Learning Outcomes

This project demonstrates practical implementation of:

* Natural Language Processing
* Machine Learning pipelines
* Text classification
* Flask backend development
* Frontend integration
* REST APIs
* Real-time prediction systems
* Data preprocessing

---

# Why This Project Stands Out

Unlike basic rule-based chatbots, this project combines:

* Machine Learning
* NLP preprocessing
* Emotion-aware interaction
* Modern UI/UX
* Real-time analytics visualization

making it a strong resume and portfolio project for AI/ML and Full Stack Development roles.

---

# Requirements

```text
Flask
scikit-learn
pandas
numpy
nltk
gunicorn
joblib
scipy
```

---

# Author

## Bhargav Hathwar

AI & Machine Learning Enthusiast

---

# License

This project is licensed under the MIT License.

---

# Show Your Support

If you liked this project:

* Star the repository
* Fork the project
* Share it with others
* Contribute improvements

---

<div alig
