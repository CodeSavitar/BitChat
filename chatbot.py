import random
import json
#from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
responses = json.loads(open('response.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')


def cleaning_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words

#print(cleaning_sentence('This is how you do a NLP Chatbot'))


def bag_of_words(sentence):
    sentence_words = cleaning_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)            

#print(bag_of_words('Hey bro what is the matter?')) 

def predict_responses(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    Error_threshold = 0.25
    res = [[i,r] for i,r in enumerate(result) if r > Error_threshold]

    res.sort(key = lambda x: x[1], reverse=True)
    return_list = []

    for r in res:
        return_list.append({'response': classes[r[0]], 'probability': str(r[1])})
    return return_list

print(predict_responses('Can you explain me a bit about Bitcoin Mining?'))



def get_response(ret_list, json_file):
    tag = ret_list[0]['response']
    list_of_responses = json_file['response']
    for i in list_of_responses:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result    

print("THE BOT IS RUNNING!!!")

while True:
    message = input("")
    resps = predict_responses(message)
    result = get_response(resps, responses)
    print(result)

