import pickle

model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

while True:
    text = input("Enter a message: ")

    vector = vectorizer.transform([text])

    emotion = model.predict(vector)[0]

    print("Detected Emotion:", emotion)