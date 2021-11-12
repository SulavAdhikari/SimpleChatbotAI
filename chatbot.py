import random, json, pickle, numpy as np

import pyttsx3
import speech_recognition as sr

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.load(open('intents.json','r'))

words = pickle.load(open('words.pk1','rb'))
classes = pickle.load(open('classes.pk1','rb'))
model = load_model('chatbotmodel.h5')

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
engine.setProperty('rate',145)

def say(text):
    print(text)
    engine.say(text)
    engine.runAndWait()

def clean_up_sentence(sentence):
    sentence_words =  nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
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
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key = lambda x: x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({
            'intent':classes[r[0]],
            'probability':str(r[1])
        })
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('Go! Bot is running')

# while True:
#     try:
#         with sr.Microphone() as source:
#             print('listening')
#             voice = listener.listen(source)
#             message = listener.recognize_google(voice)
#     except:
#         say('exiting')
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     say(res)

while True:
    try:
        message = input("YOU:")
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("BOT:",res)
    except KeyboardInterrupt:
        message ="GoodBye"
        ints=predict_class(message)
        res = get_response(ints, intents)
        print("BOT:",res)
        exit()
