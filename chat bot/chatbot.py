import nltk
import json
import pickle
import random
import tflearn
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
Facebook : https://www.facebook.com/sujit.mandal.33671748
Twitter : https://twitter.com/mandalsujit37
"""


#nltk.download()
ls = LancasterStemmer()
with open('intents.json') as file:
    data = json.load(file)

#print(data)

words = []
labels = []
words_x = []
words_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        words_x.append(wrds)
        words_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])


words = [ls.stem(w.lower()) for w in words if w not in '?']
words = sorted(list(set(words)))

labels = sorted(labels)

train = []
test = []

output = [0 for _ in range(len(labels))]

for x, docs in enumerate(words_x):
    container = []

    wrds = [ls.stem(w) for w in docs]

    for w in words:
        if w in wrds:
            container.append(1)
        else:
            container.append(0)

    target_row = output[:]
    target_row[labels.index(words_y[x])] = 1

    train.append(container)
    test.append(target_row)

train = np.array(train)
test = np.array(test)

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(test[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(train, test, n_epoch=2000, batch_size=8, show_metric=True)
model.save('model.tflearn')

def user(users, words):
    container = [0 for _ in range(len(words))]
    user_words = nltk.word_tokenize(users)
    user_words = [ls.stem(word.lower()) for word in user_words]

    for word_user in user_words:
        for i, w in enumerate(words):
            if w == word_user:
                container[i] = 1

    return(np.array(container))

def chat_boot():
    print('Start Taking With The Chat Boot (type quit to stop chat)!')
    while True:
        human = input('You : ')
        if human.lower() == 'q':
            break

        prediction = model.predict([user(human, words)])[0]
        prediction_index = np.argmax(prediction)
        tag = labels[prediction_index]
    
        acuracy = prediction[prediction_index]
    
        if acuracy > 0.7:
            for tags in data['intents']:
                if tags['tag'] == tag:
                    responses = tags['responses']

            print('Bot :',random.choice(responses))
        else:
            print("Sorry I Don't Get That! Try Again....")
     
    
chat_boot()


#OUTPUT:

'''    
    Start Taking With The Chat Boot (type quit to stop chat)!
    You : hey
    Bot : Good to see you again!
    You : hi
    Bot : Hi there, how can I help?
    You : hello
    Bot : Hi there, how can I help?
    You : how are you
    Bot : Hello!
    You : whats your name
    Bot : I'm bot
    You : how are you
    Bot : Hi there, how can I help?
    You : what is your name
    Bot : I'm bot
    You : morning
    Bot : Good Moring have a nice day
    You : good morning
    Bot : Good Moring..
    You : evening
    Bot : Good Evening..
    You : good evening
    Bot : Good Evening have a nice day
    You : night
    Bot : Good Night..
    You : good night
    Bot : Good Night have a peacfull sleep
    You : i am sad
    Bot : Try to be happy always..
    You : i am bored 
    Bot : well play some games
    You : i am getting boring
    Bot : well play some games
    You : i am sad
    Bot : Well call your bestfriend
    You : sad
    Bot : Well call your bestfriend
    You : happy
    Bot : Woow that's awesome..
    You : who is sujit
    Bot : Mandal
    You : who is sujit
    Bot : Sujit Mandal is  a student and also he is a Python Devoloper!
    You : last name of sujit
    Bot : Sujit Mandal is  a student and also he is a Python Devoloper!
    You : last name of sujit
    Bot : Mandal
    You : tell me about sujit
    Bot : Sujit Mandal is a Python Devoloper. He doing Research in Machine Learning. his github link:https://github.com/sujitmandal?tab=repositories
    You : bye
    Bot : Sad to see you go :(
    You : have a nice day
    Bot : Talk to you later
    You : hu
    Bot : well call your mother
    You : dhsagfcgsgfcs cshcsiyhcs
    Bot : well call your mother
    You : la;sajfkcka
    Bot : well call your mother
    You : fuck off
    Bot : Try to be happy always..
    You : fuck you
    Sorry I Don't Get That! Try Again....
    You : ok
    Bot : Well call your bestfriend
    You : bye
    Bot : Goodbye!
    You : q
'''