import json
import random
import numpy as np
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

#nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

responses = json.loads(open('response.json').read())

words = []
classes = []
docs = []
ignore_letters = ['?','!','.',',']

for response in responses['response']:
    for pattern in response['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        docs.append((word_list, response['tag']))
        if response not in classes:
            classes.append(response['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
#print(words)

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

train = []
empty_output = [0]*len(classes)

for doc in docs:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(empty_output)
    output_row[classes.index(doc[1])] = 1
    train.append([bag, output_row])

random.shuffle(train)
train = np.array(train)
print(type(train))

x_train = list(train[:, 0])
y_train = list(train[:, 1])

print(x_train)

model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation = 'softmax'))

sgd = SGD(lr=0.01, decay = 1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

hist = model.fit(np.array(x_train), np.array(y_train), epochs = 200, batch_size=5, verbose=1)
model.save('chatbot_model.h5',hist)
print("DONE!")