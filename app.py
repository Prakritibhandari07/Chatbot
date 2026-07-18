import json
import pickle
import random

import nltk
import numpy
import streamlit as st
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
stemmer = LancasterStemmer()


@st.cache_resource
def load_assets():
    with open("intents.json") as f:
        data = json.load(f)
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    model = tf.keras.models.load_model("chatbot_model.keras")
    return data, words, labels, model


data, words, labels, model = load_assets()


def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(s)]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array([bag])


st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for role, text in st.session_state.messages:
    st.chat_message(role).write(text)

if user_input := st.chat_input("You:"):
    st.session_state.messages.append(("user", user_input))
    st.chat_message("user").write(user_input)

    results = model.predict(bag_of_words(user_input, words))[0]
    i = numpy.argmax(results)
    tag = labels[i]

    if results[i] > 0.7:
        responses = next(t["responses"] for t in data["intents"] if t["tag"] == tag)
        reply = random.choice(responses)
    else:
        reply = "I didn't get that, try again."

    st.session_state.messages.append(("assistant", reply))
    st.chat_message("assistant").write(reply)