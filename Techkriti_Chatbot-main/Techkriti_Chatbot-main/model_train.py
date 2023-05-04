from tensorflow.python.framework import ops
import random
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import warnings
import pickle
import json
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')  # Sentence tokenizer
lemmatizer = WordNetLemmatizer()
warnings.filterwarnings('ignore')
nltk.download('wordnet')  # lexical database for the English language
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()  # read json file
intents = json.loads(data_file)  # load json file


for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)  # add each elements into list
        # combination between patterns and intents
        # add single element into end of list
        documents.append((w, intent['tag']))
        # add to tag in our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents\n", documents, "\n")
# classes = intents[tag]
print(len(classes), "classes\n", classes, "\n")
# words = all words, vocabulary
print(len(words), "unique lemmatized words\n", words, "\n")
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words
    pattern_words = doc[0]
    # convert pattern_words in lower case
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]
    # create bag of words array,if word match found in current pattern then put 1 otherwise 0.[row * colm(263)]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # in output array 0 value for each tag ang 1 value for matched tag.[row * colm(8)]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle training and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test. X - patterns(words), Y - intents(tags)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

ops.reset_default_graph()

# intialising the ANN
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compiling the ANN
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
# Fitting the ANN model to training set
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=400, batch_size=30, verbose=1)

model.save('mymodel.h5', hist)
print("model created")
