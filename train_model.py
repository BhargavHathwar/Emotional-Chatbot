import pandas as pd
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (first time only)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Keep negation and strong emotion words so the model can learn from them
KEEP_WORDS = {"not", "no", "never", "hate", "love", "don't", "didn't", "can't", "won't"}
effective_stop_words = stop_words - KEEP_WORDS


# -----------------------------
# 1 Load Dataset
# -----------------------------

data = pd.read_csv("tweet_emotions.csv")
data = data[["content", "sentiment"]]
data.columns = ["text", "emotion"]


# -----------------------------
# 2 Text Cleaning Function
# (same function used in app.py — must stay in sync)
# -----------------------------

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)                   # remove URLs
    text = re.sub(r"[^a-zA-Z\s']", "", text)              # keep only letters + apostrophes
    words = text.split()
    words = [w for w in words if w not in effective_stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


data["text"] = data["text"].apply(preprocess)
data = data[data["text"].str.strip() != ""]               # drop empty texts after cleaning


# -----------------------------
# 3 Map all 13 raw labels → 7 meaningful emotions
#
# Original labels & counts:
#   neutral(8638) worry(8459) happiness(5209) sadness(5165)
#   love(3842)  surprise(2187)  fun(1776) relief(1526)
#   hate(1323)  empty(827)  enthusiasm(759)  boredom(179)  anger(110)
#
# Design decisions:
#   • joy   = happiness + fun + enthusiasm + relief  (all clearly positive)
#   • love  = love alone  (romantic/affectionate language is distinct)
#   • sadness = sadness + empty  (emotional depletion ≈ sadness)
#   • anger = anger + hate  (anger only 110 samples; both share hostile vocabulary)
#   • worry = worry alone  (large enough, distinct anxious language)
#   • neutral = neutral + boredom  (low-affect, flat tone)
#   • surprise = surprise alone  (distinct, sufficient samples)
# -----------------------------

emotion_map = {
    "happiness":  "joy",
    "fun":        "joy",
    "enthusiasm": "joy",
    "relief":     "joy",

    "love":       "love",

    "sadness":    "sadness",
    "empty":      "sadness",

    "anger":      "anger",
    "hate":       "anger",

    "worry":      "worry",

    "neutral":    "neutral",
    "boredom":    "neutral",

    "surprise":   "surprise",
}

data["emotion"] = data["emotion"].map(emotion_map)
data = data.dropna()

print("\nEmotion Distribution after mapping:")
print(data["emotion"].value_counts())


# -----------------------------
# 4 TF-IDF Vectorization
# sublinear_tf=True reduces the weight of very common terms
# ngram_range=(1,2) captures two-word phrases like "not happy"
# -----------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)

X = vectorizer.fit_transform(data["text"])
y = data["emotion"]


# -----------------------------
# 5 Train/Test Split
# -----------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y            # preserve class ratios in both splits
)


# -----------------------------
# 6 Train Model
# C=1.0 gives best generalisation on this dataset
# -----------------------------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
model.fit(X_train, y_train)


# -----------------------------
# 7 Evaluate Model
# -----------------------------

from sklearn.metrics import accuracy_score, classification_report

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


# -----------------------------
# 8 Save Model + Artefacts
# -----------------------------

pickle.dump(model, open("emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
# Also save the stop-words set so app.py uses the identical preprocessing
pickle.dump(effective_stop_words, open("stop_words.pkl", "wb"))

print("\nModel, vectorizer, and stop_words saved successfully!")
