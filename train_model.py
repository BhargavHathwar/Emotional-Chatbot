import pandas as pd
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (first time only)
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -----------------------------
# 1 Load Dataset
# -----------------------------

data = pd.read_csv("tweet_emotions.csv")

data = data[["content","sentiment"]]

data.columns = ["text","emotion"]


# -----------------------------
# 2 Text Cleaning Function
# -----------------------------

def preprocess(text):

    text = text.lower()

    text = re.sub(r"http\S+","",text)

    text = re.sub(r"[^a-zA-Z\s]","",text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


data["text"] = data["text"].apply(preprocess)


# -----------------------------
# 3 Reduce 13 emotions → 5 emotions
# -----------------------------

emotion_map = {

"happiness":"happy",
"love":"happy",
"fun":"happy",
"enthusiasm":"happy",

"sadness":"sad",
"empty":"sad",

"worry":"stress",

"anger":"angry",
"hate":"angry",

"boredom":"neutral",
"neutral":"neutral",
"relief":"neutral",
"surprise":"neutral"

}

data["emotion"] = data["emotion"].map(emotion_map)

data = data.dropna()

print("\nEmotion Distribution:")
print(data["emotion"].value_counts())


# -----------------------------
# 4 TF-IDF Vectorization
# -----------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(data["text"])

y = data["emotion"]


# -----------------------------
# 5 Train/Test Split
# -----------------------------

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 6 Train Model
# -----------------------------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=2000)

model.fit(X_train,y_train)


# -----------------------------
# 7 Evaluate Model
# -----------------------------

from sklearn.metrics import accuracy_score,classification_report

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test,predictions)

print("\nModel Accuracy:",accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test,predictions))


# -----------------------------
# 8 Save Model
# -----------------------------

pickle.dump(model,open("emotion_model.pkl","wb"))

pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

print("\nModel and vectorizer saved successfully!")