import random
from voc import voc
from keras import layers, models, regularizers
import time
from keras.models import load_model
from spacy.lang.en import English
import numpy
from flask import Flask, render_template, request
import json
import pickle
import os
import nltk
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import warnings
import pickle
import json
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

nlp = English()
tokenizer = nlp.tokenizer
PAD_Token = 0

app = Flask(__name__)


model = load_model('mymodel.h5')

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))

classes = pickle.load(open('classes.pkl', 'rb'))

# Preprocssing of the data
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    # print(sentence_words)
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    # print(sentence_words)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    # print(bag)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
                # print ("found in bag: %s" % w) #print(bag)

    return (np.array(bag))


# 
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    # print(p)
    res = model.predict(np.array([p]))[0]
    # print(res)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # print(results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
    # print(return_list)

# Input from the user
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    # print(tag)
    list_of_intents = intents_json['intents']
    # print(list_of_intents)
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

# Responses from the Chatbot
def chatbot_response(text):
    ints = predict_class(text, model)  # print(ints)
    res = getResponse(ints, intents)
    return res


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chatbot_response(userText))


if __name__ == "__main__":
    app.run()
