"""
Module to build the GUI for the chatbot
"""

import json
import pickle
import random
from tkinter import DISABLED, END, NORMAL, TRUE, Button, Scrollbar, Text, Tk

import nltk
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

from constants import DEFAULT_RESPONSE, ERROR_THRESHOLD, CLASSES_PKL_FILE, WORDS_PKL_FILE
from train_chatbot import train_model


def get_response(ints, intents_json):
    """Fetch the random response from intents"""

    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    result = DEFAULT_RESPONSE

    for intent in list_of_intents:
        if intent["tag"] == tag:
            result = random.choice(intent["responses"])
            break

    return result


# noinspection SpellCheckingInspection
def clean_up_sentence(sentence):
    """Tokenize and lemmatize the sentence"""

    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


def bag_of_words(sentence, all_words):
    """Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence"""

    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(all_words)
    for sentence_word in sentence_words:
        for index, word in enumerate(all_words):
            if word == sentence_word:
                # assign 1 if current word is in the vocabulary position
                bag[index] = 1
                print(f"Found in bag: {word}")

    return np.array(bag)


def predict_class(sentence, class_model):
    """Predict the class model for the text input"""

    classes = pickle.load(open(CLASSES_PKL_FILE, "rb"))
    words_from_pkl = pickle.load(open(WORDS_PKL_FILE, "rb"))

    # Filter out predictions below a threshold
    bag_array = bag_of_words(sentence, words_from_pkl)
    res = class_model.predict(np.array([bag_array]))[0]

    results = [
        [index, result] for index, result in enumerate(res) if result > ERROR_THRESHOLD
    ]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)

    return [
        {"intent": classes[result[0]], "probability": str(result[1])}
        for result in results
    ]


# noinspection SpellCheckingInspection
def chatbot_response(msg):
    """Return the chatbot response"""

    model = load_model("chatbot_model.h5")
    intents = json.loads(open("intents.json").read())
    ints = predict_class(msg, model)
    response = get_response(ints, intents)
    return response


def send():
    """Send command for the UI button"""

    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("0.0", END)

    if msg != "":
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + "\n\n")
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + "\n\n")

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


if __name__ == "__main__":

    # Train the model
    train_model()

    # Creating GUI with tkinter
    base = Tk()
    base.title("ChatBot")
    base.geometry("400x500")
    base.resizable(width=TRUE, height=TRUE)

    # Create Chat window
    ChatLog = Text(
        base,
        bd=0,
        bg="grey",
        height="8",
        width="50",
        font="Arial",
    )

    ChatLog.config(state=DISABLED)

    # Bind scrollbar to Chat window
    scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
    ChatLog["yscrollcommand"] = scrollbar.set

    # Create Button to send message
    SendButton = Button(
        base,
        font=("Verdana", 12, "bold"),
        text="Send",
        width="12",
        height=5,
        bd=0,
        bg="blue",
        activebackground="#ffffff",
        fg="#ffffff",
        command=send,
    )

    # Create the box to enter message
    EntryBox = Text(base, bd=0, bg="grey", width="29", height="5", font="Arial")

    # Place all components on the screen
    scrollbar.place(x=376, y=6, height=386)
    ChatLog.place(x=6, y=6, height=386, width=370)
    EntryBox.place(x=128, y=401, height=90, width=265)
    SendButton.place(x=6, y=401, height=90)

    base.mainloop()
