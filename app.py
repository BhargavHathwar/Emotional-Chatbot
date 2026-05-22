from flask import Flask, render_template, request, jsonify
import pickle
import os
import random
import logging
import re
import threading

import nltk

# Download NLTK data to a local writable path (works on Railway, Render, etc.)
NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_DIR)
for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(_pkg, download_dir=NLTK_DATA_DIR, quiet=True)
    except Exception:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# Global model state
# ─────────────────────────────────────────────
MODEL_LOADED = False
model        = None
vectorizer   = None
effective_stop_words = None
lemmatizer   = WordNetLemmatizer()


# ─────────────────────────────────────────────
# Preprocessing — MUST match train_model.py exactly
# ─────────────────────────────────────────────
def preprocess(text):
    text  = str(text).lower()
    text  = re.sub(r"http\S+", "", text)
    text  = re.sub(r"[^a-zA-Z\s']", "", text)
    words = text.split()
    words = [w for w in words if w not in effective_stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


# ─────────────────────────────────────────────
# SYNONYM DICTIONARY
# Every word here is a trigger that maps directly to an emotion class.
# The ML model handles general language; this dictionary handles:
#   • Slang / informal words  ("stoked", "miffed", "gutted" …)
#   • Rare / literary words   ("melancholic", "elated", "irate" …)
#   • Phrases / idioms stored as single tokens after joining
#   • Negation is handled separately below
# ─────────────────────────────────────────────
SYNONYM_MAP = {

    # ── JOY / HAPPINESS ──────────────────────────────────────────────────
    # Includes verb roots so "elates", "thrilling" etc. are caught via lemmatization
    "joy": {
        "happy", "happiness", "joyful", "joyous", "cheerful", "gleeful", "merry",
        "delight", "delighted", "elate", "elated", "ecstatic", "euphoric", "overjoyed", "thrilled",
        "blissful", "bliss", "exhilarate", "exhilarated", "jubilant", "exultant", "radiant",
        "beam", "beaming", "grinning",
        # fun / playful
        "fun", "playful", "amuse", "amused", "amusing", "laugh", "laughing", "laughter", "giggling",
        # excited / enthusiastic
        "excite", "excited", "exciting", "enthusiasm", "enthusiastic", "eager",
        "energise", "energized", "pump", "pumped", "stoked", "hyped", "fired", "psyched",
        # content / satisfied
        "content", "contented", "satisfy", "satisfied", "please", "pleased",
        "grateful", "thankful", "bless", "blessed", "fortunate", "lucky",
        "relieve", "relieved", "relief",
        # positive slang
        "awesome", "amazing", "fantastic", "wonderful", "brilliant", "great",
        "incredible", "superb", "magnificent", "splendid", "terrific",
        "lit", "fire", "dope", "epic", "legendary", "goated",
        # winning / achievement
        "victorious", "triumphant", "accomplish", "accomplished", "proud",
    },

    # ── LOVE / AFFECTION ─────────────────────────────────────────────────
    "love": {
        "love", "loving", "loved", "affection", "affectionate",
        "adore", "adoring", "adoration", "cherish", "cherishing",
        "devote", "devoted", "devotion", "infatuate", "infatuated", "infatuation", "smitten",
        "enamour", "enamoured", "enamored", "enchant", "enchanted", "enchanting",
        "care", "caring", "warmth", "warm", "tender", "tenderness",
        "fond", "fondness", "attach", "attached", "attachment",
        "romantic", "romance", "passion", "passionate",
        "crush", "crushing", "attract", "attracted", "attraction",
        "compassion", "compassionate", "empathy", "empathetic",
        "nurture", "nurturing",
        "obsess", "obsessed", "obsession",
    },

    # ── SADNESS / GRIEF ───────────────────────────────────────────────────
    "sadness": {
        "sad", "sadness", "unhappy", "sorrowful", "sorrow",
        "grief", "grieve", "grieving", "mourn", "mourning",
        "heartbreak", "heartbroken", "broken",
        "devastate", "devastated", "devastation", "crush", "crushed", "shatter", "shattered",
        "depress", "depressed", "depression", "melancholy", "melancholic",
        "miserable", "misery", "gloomy", "gloom", "dejected", "dejection",
        "despondent", "despondency", "hopeless", "hopelessness",
        "empty", "emptiness", "numb", "numbness", "hollow",
        "worthless", "worthlessness", "lose", "lost", "alone", "lonely", "loneliness",
        "cry", "crying", "weep", "weeping", "wept", "tears", "sob", "sobbing",
        "hurt", "hurting", "pain", "painful", "ache", "aching",
        "wound", "wounded", "suffer", "suffering",
        "disappoint", "disappointed", "disappointment",
        "disillusion", "disillusioned", "dishearten", "disheartened",
        "gutted", "bummed", "miss", "missing", "longing", "homesick",
        "down", "blue", "low", "rough", "rotten", "awful",
    },

    # ── WORRY / ANXIETY / FEAR ────────────────────────────────────────────
    "worry": {
        "worry", "worried", "worrying", "anxious", "anxiety",
        "nervous", "nervousness", "stress", "stressed", "stressful",
        "tense", "tension", "uneasy", "unease",
        "afraid", "fear", "fearful", "scared",
        "frighten", "frightened", "fright",
        "terrify", "terrified", "terror",
        "horrify", "horrified", "horror",
        "petrify", "petrified",
        "dread", "dreading", "dreaded",
        "panic", "panicking", "panicked", "frantic",
        "overwhelm", "overwhelmed", "overwhelming",
        "concern", "concerned", "apprehensive", "apprehension",
        "hesitant", "hesitation", "doubtful", "doubt", "uncertain", "uncertainty",
        "insecure", "insecurity", "vulnerable",
        "shake", "shaking", "tremble", "trembling", "sweat", "sweating",
        "paranoid", "paranoia", "freaking", "spiral", "spiraling",
    },

    # ── ANGER / HATE / DISGUST ────────────────────────────────────────────
    "anger": {
        "angry", "anger", "mad", "furious", "fury", "rage", "raging",
        "enrage", "enraged", "irate", "livid",
        "seethe", "seething", "fume", "fuming",
        "outrage", "outraged", "indignant", "indignation",
        "wrath", "wrathful", "incense", "incensed",
        "hate", "hatred", "hateful", "hating",
        "loathe", "loathing", "despise", "despising",
        "detest", "detesting", "abhor", "abhorring",
        "disgust", "disgusted", "revolt", "revolted", "revulsion",
        "repulse", "repulsed",
        "contempt", "contemptuous", "scorn", "scornful",
        "frustrate", "frustrated", "frustration",
        "irritate", "irritated", "irritation",
        "annoy", "annoyed", "annoying",
        "aggravate", "aggravated", "aggravation",
        "exasperate", "exasperated", "exasperation",
        "infuriate", "infuriated", "infuriation",
        "bitter", "bitterness", "resent", "resentful", "resentment",
        "hostile", "hostility", "aggressive", "aggression",
        "pissed", "ticked", "steaming", "boiling", "salty", "triggered",
    },

    # ── SURPRISE / SHOCK ──────────────────────────────────────────────────
    "surprise": {
        "surprise", "surprised", "shock", "shocked", "shocking",
        "astonish", "astonished", "astonishment",
        "astound", "astounded", "astounding",
        "amaze", "amazed", "amazement",
        "stun", "stunned", "stunning",
        "bewilder", "bewildered", "bewilderment",
        "baffle", "baffled", "bafflement",
        "flabbergasted", "gobsmacked", "speechless",
        "disbelief", "unbelievable", "unexpected", "incredible",
        "mindblowing", "mindblown",
        "awestruck", "awe", "wonder", "wonderstruck",
        "whoa", "woah", "omg", "omigosh", "whaaat",
    },

    # ── NEUTRAL / BOREDOM ─────────────────────────────────────────────────
    "neutral": {
        "neutral", "okay", "ok", "fine", "alright", "average", "normal", "ordinary",
        "bore", "bored", "boredom", "boring", "dull", "dullness", "tedious", "tedium",
        "monotonous", "monotony", "uneventful", "uninteresting",
        "tire", "tired", "tiredness", "exhaust", "exhausted", "exhaustion", "sleepy",
        "indifferent", "indifference", "apathetic", "apathy",
        "meh", "whatever", "unbothered",
        "calm", "serene", "serenity", "peaceful", "peace",
        "relax", "relaxed", "relaxation", "chill", "chilling",
        "unmotivated", "uninspired", "blank", "flat",
    },
}

# Build a flat lookup: word → emotion  (used for O(1) matching)
WORD_TO_EMOTION: dict[str, str] = {}
for emotion, words in SYNONYM_MAP.items():
    for word in words:
        WORD_TO_EMOTION[word.lower()] = emotion


# ─────────────────────────────────────────────
# NEGATION tokens
# ─────────────────────────────────────────────
NEGATIONS = {
    "not", "no", "never", "don't", "dont", "didn't", "didnt",
    "isn't", "isnt", "wasn't", "wasnt", "can't", "cant",
    "won't", "wont", "neither", "nor", "barely", "hardly", "scarcely",
}


# ─────────────────────────────────────────────
# Confidence threshold below which we say "not found"
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 28   # percent; below this → unknown


# ─────────────────────────────────────────────
# Responses (multiple per emotion for variety)
# ─────────────────────────────────────────────
responses = {
    "joy": [
        "That's wonderful! 😊 Your happiness is contagious — keep riding that wave!",
        "Love the good vibes! 🌟 What's making today so great for you?",
        "You're glowing with positivity! 😄 Enjoy every bit of it!",
        "That sounds like a fantastic feeling! 🎉 Share the joy — what happened?",
    ],
    "love": [
        "Aww, that's so sweet! 💕 Love is a beautiful feeling.",
        "It sounds like someone (or something) has a special place in your heart. 💖",
        "Love is in the air! 🌹 Those feelings are something to cherish.",
        "That warmth and affection you're feeling? Hold onto it. 💞",
    ],
    "sadness": [
        "I'm really sorry you're feeling this way. 💙 It's okay to feel sad — emotions need space.",
        "That sounds heavy. 🌧️ I'm here if you want to talk it through.",
        "Sending you warmth. 💙 You're stronger than you know, even when it doesn't feel that way.",
        "It's okay not to be okay. 🌊 Allow yourself to feel it — better days are ahead.",
    ],
    "worry": [
        "Sounds like you have a lot on your mind. 😟 Take a slow, deep breath — one thing at a time.",
        "Anxiety can feel overwhelming. 🌿 Is there one small step you can take right now?",
        "You've navigated tough moments before — you've got this. 💪 What's worrying you most?",
        "It's natural to worry. 🧘 Try grounding yourself: name 5 things you can see around you.",
    ],
    "anger": [
        "I hear you — that frustration is completely valid. 😠 Take a moment to breathe before reacting.",
        "Anger tells you something matters to you. 🔥 What's really behind the frustration?",
        "It's okay to feel angry. 😤 Step back for a bit and let the heat settle.",
        "That sounds genuinely infuriating. 💢 Venting can help — what happened?",
    ],
    "surprise": [
        "Whoa, sounds unexpected! 😲 Good surprise or not-so-good?",
        "Life loves throwing curveballs! 🎲 How are you feeling about it?",
        "That's quite a development! 😮 Tell me more — I'm all ears.",
        "Sometimes the unexpected changes everything. 🌀 How are you processing it?",
    ],
    "neutral": [
        "I see. Tell me more about what's on your mind. 🙂",
        "Sounds like a calm moment. 😌 Anything specific you'd like to talk about?",
        "Got it. I'm here whenever you want to share more. 🙂",
        "Sometimes a quiet moment is exactly what we need. 🍃 How's your day going overall?",
    ],
}

UNKNOWN_REPLIES = [
    "Hmm, I wasn't able to identify a specific emotion in that. 🤔 Could you describe how you're feeling in a different way?",
    "I couldn't quite detect an emotion there. 😕 Try sharing a bit more about how you're feeling?",
    "That one slipped past me — emotion could not be identified. 🧩 Can you rephrase or add more detail?",
    "I'm not sure what emotion that reflects. 🔍 Feel free to describe your feelings more directly!",
]


# ─────────────────────────────────────────────
# Synonym lookup (negation-aware, multi-word aware)
# Returns emotion string or None
# ─────────────────────────────────────────────
def synonym_lookup(text: str) -> str | None:
    """
    Scan the message for any synonym from WORD_TO_EMOTION.
    Checks single words AND two-word phrases.
    Skips any match that is preceded by a negation word.
    Returns the matched emotion, or None if no synonym found.
    """
    clean = re.sub(r"[^a-zA-Z\s']", " ", text.lower())
    words = clean.split()

    for i, word in enumerate(words):
        prev = words[i - 1] if i > 0 else ""

        # Skip if negated
        if prev in NEGATIONS:
            continue

        # Single-word match — try raw, noun-lemma, and verb-lemma forms
        lemma_n = lemmatizer.lemmatize(word, pos="n")   # default (noun)
        lemma_v = lemmatizer.lemmatize(word, pos="v")   # verb form (e.g. cherishes→cherish)
        for form in (word, lemma_n, lemma_v):
            if form in WORD_TO_EMOTION:
                return WORD_TO_EMOTION[form]

        # Two-word phrase match (e.g. "head over" won't match, but "broken heart" etc.)
        if i < len(words) - 1:
            bigram = word + " " + words[i + 1]
            if bigram in WORD_TO_EMOTION:
                return WORD_TO_EMOTION[bigram]

    return None


# ─────────────────────────────────────────────
# Train and save model (fallback if pkl missing)
# ─────────────────────────────────────────────
def train_and_save():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    logger.info("Training model from scratch (memory-optimised)...")

    _stop_words = set(stopwords.words("english"))
    _keep       = {"not", "no", "never", "hate", "love", "don't", "didn't", "can't", "won't"}
    _eff_stop   = _stop_words - _keep

    # Simple preprocess — no lemmatization to save RAM & time
    def _preprocess(text):
        text  = str(text).lower()
        text  = re.sub(r"http\S+", "", text)
        text  = re.sub(r"[^a-zA-Z\s']", "", text)
        words = [w for w in text.split() if w not in _eff_stop]
        return " ".join(words)

    data = pd.read_csv(os.path.join(BASE_DIR, "tweet_emotions.csv"))
    data = data[["content", "sentiment"]]
    data.columns = ["text", "emotion"]

    emotion_map = {
        "happiness": "joy", "fun": "joy", "enthusiasm": "joy", "relief": "joy",
        "love":      "love",
        "sadness":   "sadness", "empty": "sadness",
        "anger":     "anger",   "hate":  "anger",
        "worry":     "worry",
        "neutral":   "neutral", "boredom": "neutral",
        "surprise":  "surprise",
    }
    data["emotion"] = data["emotion"].map(emotion_map)
    data = data.dropna()

    # Sample max 15k rows per class to keep memory under 512 MB
    data = data.groupby("emotion", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 2000), random_state=42)
    ).reset_index(drop=True)

    data["text"] = data["text"].apply(_preprocess)
    data = data[data["text"].str.strip() != ""]

    logger.info(f"Training on {len(data)} rows across {data['emotion'].nunique()} emotions")

    # Unigrams only + smaller vocab to keep sparse matrix small
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), sublinear_tf=True, min_df=2)
    X   = vec.fit_transform(data["text"])
    y   = data["emotion"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mdl = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial")
    mdl.fit(X_train, y_train)

    acc = mdl.score(X_test, y_test)
    logger.info(f"Model accuracy: {acc:.2%}")

    pickle.dump(mdl, open(os.path.join(BASE_DIR, "emotion_model.pkl"), "wb"))
    pickle.dump(vec, open(os.path.join(BASE_DIR, "vectorizer.pkl"),    "wb"))
    pickle.dump(_eff_stop, open(os.path.join(BASE_DIR, "stop_words.pkl"), "wb"))
    logger.info("Model trained and saved.")
    return mdl, vec, _eff_stop


# ─────────────────────────────────────────────
# Load or retrain in background thread
# ─────────────────────────────────────────────
def _sklearn_version_mismatch():
    """Return True if the saved pkl was built with a different sklearn version."""
    import sklearn
    model_path = os.path.join(BASE_DIR, "emotion_model.pkl")
    try:
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tmp = pickle.load(open(model_path, "rb"))
            for w in caught:
                if "InconsistentVersionWarning" in str(w.category.__name__):
                    logger.warning("sklearn version mismatch detected — will retrain.")
                    return True
    except Exception:
        return True
    return False


def load_or_train():
    global model, vectorizer, effective_stop_words, MODEL_LOADED
    try:
        model_path      = os.path.join(BASE_DIR, "emotion_model.pkl")
        vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
        stop_words_path = os.path.join(BASE_DIR, "stop_words.pkl")

        # Force retrain if pkl was built with a different sklearn version
        if _sklearn_version_mismatch():
            raise ValueError("sklearn version mismatch — retraining for compatibility")

        model      = pickle.load(open(model_path,      "rb"))
        vectorizer = pickle.load(open(vectorizer_path, "rb"))

        if os.path.exists(stop_words_path):
            effective_stop_words = pickle.load(open(stop_words_path, "rb"))
        else:
            _base = set(stopwords.words("english"))
            _keep = {"not", "no", "never", "hate", "love", "don't", "didn't", "can't", "won't"}
            effective_stop_words = _base - _keep

        vectorizer.transform(["test"])
        model.predict(vectorizer.transform(["test"]))
        logger.info("Model loaded successfully.")
        MODEL_LOADED = True

    except Exception as e:
        logger.warning(f"Model load failed ({e}), retraining...")
        try:
            model, vectorizer, effective_stop_words = train_and_save()
            MODEL_LOADED = True
            logger.info("Retraining complete.")
        except Exception as e2:
            logger.error(f"Retraining failed: {e2}")
            MODEL_LOADED = False


# Start model load in background thread so gunicorn starts immediately.
logger.info("Scheduling model load in background thread...")
threading.Thread(target=load_or_train, daemon=True).start()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok" if MODEL_LOADED else "loading", "model_loaded": MODEL_LOADED})


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model is still loading — please wait 30 seconds and try again."}), 503

    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = str(data["message"]).strip()
    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        # ── STEP 1: Synonym dictionary lookup (highest priority) ──────────
        synonym_emotion = synonym_lookup(message)

        # ── STEP 2: ML model prediction ───────────────────────────────────
        processed  = preprocess(message)
        vector     = vectorizer.transform([processed])
        ml_emotion = model.predict(vector)[0]
        proba      = model.predict_proba(vector)[0]
        classes    = model.classes_
        confidence = round(float(max(proba)) * 100)

        top2 = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:2]
        top2_result = [{"emotion": e, "confidence": round(float(c) * 100)} for e, c in top2]

        # ── STEP 3: Decide final emotion ──────────────────────────────────
        #  Priority order:
        #    1. Synonym dict hit  → use it (explicit word found)
        #    2. ML confident      → use ML result
        #    3. Neither           → "unknown"
        if synonym_emotion:
            final_emotion = synonym_emotion
            detection_method = "synonym"
        elif confidence >= CONFIDENCE_THRESHOLD:
            final_emotion = ml_emotion
            detection_method = "ml_model"
        else:
            # Could not determine emotion with enough certainty
            logger.info(f"Unknown emotion | conf={confidence}% | Input: {message[:60]}")
            return jsonify({
                "emotion":    "unknown",
                "confidence": confidence,
                "top2":       top2_result,
                "reply":      random.choice(UNKNOWN_REPLIES),
                "detection":  "none",
            })

        reply = random.choice(responses.get(final_emotion, [random.choice(UNKNOWN_REPLIES)]))

        logger.info(
            f"[{detection_method}] Predicted: {final_emotion} ({confidence}%) | Input: {message[:60]}"
        )

        return jsonify({
            "emotion":    final_emotion,
            "confidence": confidence,
            "top2":       top2_result,
            "reply":      reply,
            "detection":  detection_method,
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed. Please try again."}), 500


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
