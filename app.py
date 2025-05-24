from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
import datetime
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_simplilearnmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure I understand. Could you rephrase?"
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "").lower()

    if "how are you" in msg:
        return jsonify({"reply": "I am fine, what about you?"})

    feelings = ["i am fine", "i'm good", "feeling good", "i am good", "i feel good"]
    if any(feeling in msg for feeling in feelings):
        return jsonify({"reply": "Glad to hear that! How can I assist you today?"})

    if "time" in msg:
        now = datetime.datetime.now()
        return jsonify({"reply": f"The current time is {now.strftime('%H:%M:%S')}"})

    if "day" in msg or "date" in msg:
        now = datetime.datetime.now()
        return jsonify({"reply": f"Today is {now.strftime('%A, %B %d, %Y')}"})

    intents_predicted = predict_class(msg)
    res = get_response(intents_predicted, intents)
    return jsonify({"reply": res})

if __name__ == "__main__":
    app.run(debug=True)
