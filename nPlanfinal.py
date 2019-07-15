# -*- coding: utf-8 -*-
"""
Created on Tuesday June 06 12:05:47 2019
@author: Dr Clement Etienam
nPlan Take home test
"""
from __future__ import print_function
print(__doc__)

from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential 
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM
from keras.layers import Embedding
import re
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from nltk import word_tokenize
import nltk; nltk.download('punkt')
from collections import Counter
import operator
from numpy.random import choice



## This section is to prevent Windows from sleeping when executing the Python script
class WindowsInhibitor:
    '''Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'''
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

    def inhibit(self):
        import ctypes
        #Preventing Windows from going to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS | \
            WindowsInhibitor.ES_SYSTEM_REQUIRED)

    def uninhibit(self):
        import ctypes
        #Allowing Windows to go to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS)


osSleep = None
# in Windows, prevent the OS from sleeping while we run
if os.name == 'nt':
    osSleep = WindowsInhibitor()
    osSleep.inhibit()
print('Start of code')
# Plot / Graph stuffs
print('Question 1-Exploring the data')
init_data = pd.read_csv("winemag-data_first150k.csv")
clement=init_data
clement.to_csv(index=False)

print("Length of dataframe before duplicates are removed:", len(init_data))
init_data.head()
parsed_data = init_data[init_data.duplicated('description', keep=False)]
print("Length of dataframe after duplicates are removed:", len(parsed_data))

parsed_data.dropna(subset=['description', 'points'])
print("Length of dataframe after NaNs are removed:", len(parsed_data))

parsed_data.head()

dp = parsed_data[['description','points']]
dp.info()
dp.head()
print('')
fig, ax = plt.subplots(figsize=(8,8))
plt.xticks(fontsize=8) # X Ticks
plt.yticks(fontsize=8) # Y Ticks
ax.set_title('Number of wines per points', fontweight="bold", size=8) # Title
ax.set_ylabel('Number of wines', fontsize = 8) # Y label
ax.set_xlabel('Points', fontsize = 8) # X label
dp.groupby(['points']).count()['description'].plot(ax=ax, kind='bar')

print('')
dp = dp.assign(description_length = dp['description'].apply(len))
dp.info()
dp.head()

print('')
fig, ax = plt.subplots(figsize=(8,8))
sns.boxplot(x='points', y='description_length', data=dp)
plt.xticks(fontsize=8) # X Ticks
plt.yticks(fontsize=8) # Y Ticks
ax.set_title('Description Length per Points', fontweight="bold", size=8) # Title
ax.set_ylabel('Description Length', fontsize = 8) # Y label
ax.set_xlabel('Points', fontsize = 8) # X label
plt.show()

#Transform method taking points as param
def transform_points_simplified(points):
    if points < 84:
        return 1
    elif points >= 84 and points < 88:
        return 2 
    elif points >= 88 and points < 93:
        return 3 
    elif points >= 93 and points < 96:
        return 4 
    else:
        return 5
dp = dp.assign(points_simplified = dp['points'].apply(transform_points_simplified))
dp.head()
fig, ax = plt.subplots(figsize=(8,8))
plt.xticks(fontsize=8) # X Ticks
plt.yticks(fontsize=8) # Y Ticks
ax.set_title('Number of wines per points', fontweight="bold", size=8) # Title
ax.set_ylabel('Number of wines', fontsize = 8) # Y label
ax.set_xlabel('Points', fontsize = 8) # X label
dp.groupby(['points_simplified']).count()['description'].plot(ax=ax, kind='bar')

fig, ax = plt.subplots(figsize=(8,8))
sns.boxplot(x='points_simplified', y='description_length', data=dp)
plt.xticks(fontsize=8) # X Ticks
plt.yticks(fontsize=8) # Y Ticks
ax.set_title('Description Length per Points', fontweight="bold", size=8) # Title
ax.set_ylabel('Description Length', fontsize = 8) # Y label
ax.set_xlabel('Points', fontsize = 8) # X label
plt.show()
print('')
print('start text clasification of the data')
X = dp['description']
y = dp['points_simplified']

vectorizer = CountVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)
# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
print('Density: {}'.format((density)))
# Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
rfc = RandomForestClassifier(n_estimators=500,n_jobs=-1)
rfc.fit(X_train, y_train)

# Testing the model
predictions = rfc.predict(X_test)
ff=accuracy_score(y_test, predictions)*100
print('The accuracy of RandomForest and Countvectorizer is',ff)
cm = confusion_matrix(y_test,predictions,labels=rfc.classes_)
print(classification_report(y_test, predictions))
print("Confusion matrix after RandomForest and Countvectorizer model")
print(cm)
print('')
# ---- Decision Tree -----------
from sklearn import tree
rfc= tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
rfc.fit(X_train, y_train)

# Testing the model
predictions = rfc.predict(X_test)
ff=accuracy_score(y_test, predictions)*100
print('The accuracy of Decisiontreeclassifier and Countvectorizer is',ff)
cm = confusion_matrix(y_test,predictions,labels=rfc.classes_)
print(classification_report(y_test, predictions))
print("Confusion matrix after Decisiontreeclasifier and Countvectorizer model")
print(cm)
print('')

from xgboost import XGBClassifier
rfc = XGBClassifier()
rfc.fit(X_train, y_train)

# Testing the model
predictions = rfc.predict(X_test)
ff=accuracy_score(y_test, predictions)*100
print('The accuracy of xgboost and Countvectorizer is',ff)
cm = confusion_matrix(y_test,predictions,labels=rfc.classes_)
print(classification_report(y_test, predictions))
print("Confusion matrix after xgboost and Countvectorizer model")
print(cm)
print('')

# -------- Nearest Neighbors ----------
from sklearn import neighbors
rfc = neighbors.KNeighborsClassifier()
rfc.fit(X_train, y_train)

# Testing the model
predictions = rfc.predict(X_test)
ff=accuracy_score(y_test, predictions)*100
print('The accuracy of Nearestneighbour and Countvectorizer is',ff)
cm = confusion_matrix(y_test,predictions,labels=rfc.classes_)
print(classification_report(y_test, predictions))
print("Confusion matrix after Nearestneighbour and Countvectorizer model")
print(cm)
print('')
# ---------- SGD Classifier -----------------
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

rfc = OneVsRestClassifier(SGDClassifier())
rfc.fit(X_train, y_train)

# Testing the model
predictions = rfc.predict(X_test)
ff=accuracy_score(y_test, predictions)*100
print('The accuracy of SGD and Countvectorizer is',ff)
cm = confusion_matrix(y_test,predictions,labels=rfc.classes_)
print(classification_report(y_test, predictions))
print("Confusion matrix after SGD and Countvectorizer model")
print(cm)
print('')

print('')
# ----------- Neural network - Multi-layer Perceptron  ------------
rfc = MLPClassifier(solver= 'lbfgs',max_iter=3000)
rfc.fit(X_train, y_train)

# Testing the model
predictions = rfc.predict(X_test)
ff=accuracy_score(y_test, predictions)*100
print('The accuracy of NN and Countvectorizer is',ff)
cm = confusion_matrix(y_test,predictions,labels=rfc.classes_)
print(classification_report(y_test, predictions))
print("Confusion matrix after NN and Countvectorizer model")
print(cm)
print('')
# --------- Logistic Regression ---------
from sklearn.linear_model import LogisticRegression
rfc = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1000)

rfc.fit(X_train, y_train)
# Testing the model
predictions = rfc.predict(X_test)
ff=accuracy_score(y_test, predictions)*100
print('The accuracy of Logisticregression and Countvectorizer is',ff)
cm = confusion_matrix(y_test,predictions,labels=rfc.classes_)
print(classification_report(y_test, predictions))
print("Confusion matrix after Logisticregression and Countvectorizer model")
print(cm)


print('')
print('Use Tfidfvectorizer')
X = dp['description']
y = dp['points_simplified']

# Vectorizing model
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
# Training model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
rfc = RandomForestClassifier(n_estimators=500,n_jobs=-1)
rfc.fit(X_train, y_train)

# Testing model
predictions = rfc.predict(X_test)
print(classification_report(y_test, predictions))    
ff=accuracy_score(y_test, predictions)*100
print('The accuracy of RandomForest and Tfidfvectorizer is',ff)
cm = confusion_matrix(y_test,predictions,labels=rfc.classes_)
print(classification_report(y_test, predictions))
print("Confusion matrix after RandomForest and Tfidfvectorizer model")
print(cm)
print('')

rfc = XGBClassifier()
rfc.fit(X_train, y_train)
# Testing the model
predictions = rfc.predict(X_test)
ff=accuracy_score(y_test, predictions)*100
print('The accuracy of xgboost and Tfidfvectorizer is',ff)
cm = confusion_matrix(y_test,predictions,labels=rfc.classes_)
print(classification_report(y_test, predictions))
print("Confusion matrix after xgboost and Tfidfvectorizer model")
print(cm)
print('')
rfc = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1000)

rfc.fit(X_train, y_train)
# Testing the model
predictions = rfc.predict(X_test)
ff=accuracy_score(y_test, predictions)*100
print('The accuracy of Logisticregression and Tfidfvectorizer is',ff)
cm = confusion_matrix(y_test,predictions,labels=rfc.classes_)
print(classification_report(y_test, predictions))
print("Confusion matrix after Logisticregression and Tfidfvectorizer model")
print(cm)

print('')
rfc = MLPClassifier(solver= 'lbfgs',max_iter=3000)
rfc.fit(X_train, y_train)

# Testing the model
predictions = rfc.predict(X_test)
ff=accuracy_score(y_test, predictions)*100
print('The accuracy of NN and Tfidfvectorizer is',ff)
cm = confusion_matrix(y_test,predictions,labels=rfc.classes_)
print(classification_report(y_test, predictions))
print("Confusion matrix after NN and Tfidfvectorizer model")
print(cm)
print('')

print('Question 2(b)')

# Main settings
epochs = 20
embedding_dim = 50
maxlen = 100
output_file = 'data/output.txt'
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
# Run grid search for each source (yelp, amazon, imdb)

print('Running grid search for data set :')
df = pd.read_csv('winemag-data_first150k.csv')
sentences = df['description'].values
y = df['points'].values
def transform_points_simplified(points):
    if points< 84:
        return 1
    elif points >= 84 and points < 88:
        return 2 
    elif points>= 88 and points < 92:
        return 3 
    elif points>= 92 and points < 96:
        return 4 
    else:
        return 5
y2=np.ravel(np.zeros((150930,1)))
for i in  range(150930):
    y2[i]=transform_points_simplified(y[i])

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
# Train-test split
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y2, test_size=0.25, random_state=1000)

# Tokenize words
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

# Pad sequences with zeros
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Parameter grid for grid search
param_grid = dict(num_filters=[32, 64, 128],
                  kernel_size=[3, 5, 7],
                  vocab_size=[vocab_size],
                  embedding_dim=[embedding_dim],
                  maxlen=[maxlen])
model = KerasClassifier(build_fn=create_model,
                        epochs=epochs, batch_size=10,
                        verbose=False)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,cv=4, verbose=1, n_iter=5)

y_train = to_categorical(y_train)
grid_result = grid.fit(X_train, y_train)
plot_history(grid_result)
# Evaluate testing set
testingvalue=grid.predict(X_test)
testingvalue=np.argmax(testingvalue, axis=-1) 
#test_accuracy = grid.score(X_test, y_test)
from sklearn.metrics import accuracy_score
test_accuracy =accuracy_score(y_test, testingvalue)*100
# Save and evaluate results
source='wine_data-set'
with open(output_file, 'a') as f:
    s = ('Running {} data set\nBest Accuracy : '
         '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
    output_string = s.format(
        source,
        grid_result.best_score_,
        grid_result.best_params_,
        test_accuracy)
    print(output_string)
    f.write(output_string)

print('Question 3(b)-Langauge model text classification and generation')
# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text
import pandas as pd
# source text

init_data = pd.read_csv("winemag-data_first150k.csv")
clement=init_data

clement["description"]= clement["description"].astype(str) 
data2=clement["description"]
data3=data2

data4=pd.Series.to_string(data2)

for i in range (20):
    freal=open("realtext.dat", "a+")
    fl=open("modellinesequence.dat", "a+")
    f2in=open("model2in1.dat", "a+")
    data=data3[i]
    rgx = re.compile("(\w[\w']*\w|\w)")
    out=rgx.findall(data)
    first=out[0]
    valuee=len(data.split())
    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])
    encoded = tokenizer.texts_to_sequences([data])[0]
    # retrieve vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # encode 2 words -> 1 word
    sequences = list()
    for i in range(1, len(encoded)):
    	sequence = encoded[i-1:i+1]
    	sequences.append(sequence)
    print('Total Sequences: %d' % len(sequences))
    # pad sequences
    max_length = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
    print('Max Sequence Length: %d' % max_length)
    # split into input and output elements
    sequences = array(sequences)
    X, y = sequences[:,:-1],sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size)
    
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 30, input_length=max_length-1))
    model.add(LSTM(60))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    # compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(X, y, epochs=500, verbose=2)
    # evaluate model
    
    
    # prepare the tokenizer on the source text
    tokenizer2 = Tokenizer()
    tokenizer2.fit_on_texts([data])
    # determine the vocabulary size
    vocab_size = len(tokenizer2.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # create line-based sequences
    sequences = list()
    for line in data.split('\n'):
    	encoded = tokenizer2.texts_to_sequences([line])[0]
    	for i in range(1, len(encoded)):
    		sequence = encoded[:i+1]
    		sequences.append(sequence)
    print('Total Sequences: %d' % len(sequences))
    # pad input sequences
    max_length2 = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_length2, padding='pre')
    print('Max Sequence Length: %d' % max_length2)
    # split into input and output elements
    sequences = array(sequences)
    X, y = sequences[:,:-1],sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size)
    # define model
    model2 = Sequential()
    model2.add(Embedding(vocab_size, 10, input_length=max_length2-1))
    model2.add(LSTM(50))
    model2.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    # compile network
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model2.fit(X, y, epochs=500, verbose=2)
    # evaluate model
    print(generate_seq(model2, tokenizer2, max_length2-1, first, valuee))
    text1=generate_seq(model2, tokenizer2, max_length2-1, first, valuee)
#    text_file.write("Purchase Amount: %s" % TotalAmount)
    fl.write("%s\n" %text1)
    #fl.write("%s\r\n" %text1)
    print(generate_seq(model, tokenizer, max_length-1, first, valuee))
    text2=generate_seq(model, tokenizer, max_length-1, first, valuee)
    f2in.write("%s\n" %text2 )
    freal.write("%s\n" %data)
    fl.close()
    f2in.close()
    freal.close()

print('')
print('Question 3-Trigram Model for text classification') 

class sentence:
    def __init__(self):
        self.unigrams = []
        self.bigrams = []
        self.trigrams = []
        
    def createNgrams(self, sent):
        # Create Unigrams
        self.unigrams = sent.split()
        
        # Create Bigrams
        for i in range(len(self.unigrams)-1):
            if self.unigrams[i] != '</s>':
                self.bigrams.append(self.unigrams[i]+' '+self.unigrams[i+1])
            
        # Create Trigrams
        for i in range(len(self.unigrams)-2):
            if self.unigrams[i] != '</s>':
                if self.unigrams[i+1] != '</s>':
                    self.trigrams.append(self.unigrams[i]+' '+self.unigrams[i+1]+' '+self.unigrams[i+2])


def preprocess_data(df):
    train_data = []
    
    for i in range(len(df)):
        if df.description[i] != 'description':
#        if df.description[i] != 'text':
            sent = '<s> <s>'
            temp = df.description[i].lower()
#            temp = re.sub(r'\<.*\>', '', temp)
#            temp = temp.replace('/', '').replace('{a', '').replace('{c', '').replace('{d', '').replace('{e', '').replace('{f', '').replace('}', '').replace('[', '').replace(']', '').replace('+', '').replace('#', '').replace('(', '').replace(')', '').replace('--', '').replace('-','')
            tokentemp = word_tokenize(temp)
            for i in range(len(tokentemp)):
                sent+=' '+tokentemp[i]
            sent+=' </s>'
        train_data.append(sent)
    return(train_data)

def get_bigrams(sent):
    a = sentence()
    a.createNgrams(sent)
    return(a.bigrams)
    
def get_trigrams(sent):
    a = sentence()
    a.createNgrams(sent)
    return(a.trigrams)


def calc_add1_smoothing_trigram(testsent):
    global prob_add1_trigram
    global trigrams_count
    global bigrams_count
    global vocab
    global generate_sent_map_trigram
    
    trigrams_list = get_trigrams(testsent)
    
    for i in range(len(trigrams_list)):
        
        b = trigrams_list[i].split()
        key = b[2]+'|'+b[0]+' '+b[1]
        value = (trigrams_count[trigrams_list[i]]+1)/(bigrams_count[b[0]+' '+b[1]]+vocab)
        
        prob_add1_trigram[key] = value
        
        temp = b[0]+' '+b[1]
        if temp in generate_sent_map_trigram.keys():
            generate_sent_map_trigram[temp].add(b[2])
        else:
            generate_sent_map_trigram[temp] = set({b[2]})


def calc_add1_smoothing_bigram(testsent):
    global prob_add1_bigram
    global unigrams_count
    global bigrams_count
    global vocab
    global generate_sent_map_bigram
    
    bigrams_list = get_bigrams(testsent)
    for i in range(len(bigrams_list)):
        
        b = bigrams_list[i].split()
        key = b[1]+'|'+b[0]
        value = (bigrams_count[bigrams_list[i]]+1)/(unigrams_count[b[0]]+vocab)
        
        prob_add1_bigram[key] = value
        
        temp = b[0]
        if temp in generate_sent_map_bigram.keys():
            generate_sent_map_bigram[temp].add(b[1])
        else:
            generate_sent_map_bigram[temp] = set({b[1]})


def calc_add1_smoothing_unigram(testsent):
    global prob_add1_unigram
    global unigrams_count
    global vocab
    global c
    
    unigrams_list = testsent.split()
    for i in range(len(unigrams_list)):
        
        key = unigrams_list[i]
        value = (unigrams_count[unigrams_list[i]]+1)/(c+vocab)
        prob_add1_unigram[key] = value


def calc_perplexity_add1(testsent):
    global prob_add1_trigram
    global trigrams_count
    global bigrams_count
    global vocab
    
    trigrams_list = get_trigrams(testsent)
    
    temp = 1
    for i in range(len(trigrams_list)):
        b = trigrams_list[i].split()
        key = b[2]+'|'+b[0]+' '+b[1]

        if key not in prob_add1_trigram:
            val = (0+1)/(bigrams_count[b[0]+' '+b[1]]+vocab)
        else:
            val = prob_add1_trigram[key]
        
        temp = temp / val

    perplexity = temp**(1/len(trigrams_list))
    return(perplexity)


def predict_next_word_interpolation(bigram):
    
    elements = []
    weights = []
    for i in generate_sent_map_interpolation[bigram]:
        elements.append(i)
        weights.append(prob_interpolation[i+'|'+bigram])

    
    weights=np.array(weights)
    weights /= weights.sum()
    nextword = choice(elements, p = weights)
    return(nextword)

    
def generate_sent_interpolation():
    
    bigram = '<s> <s>'
    final_sent = '<s> '
    next_word = predict_next_word_interpolation(bigram)
    bigram = bigram.split()[1]+' '+next_word
    final_sent+=next_word
    while bigram.split()[1] != '</s>':
        next_word = predict_next_word_interpolation(bigram)
        bigram = bigram.split()[1]+' '+next_word
        final_sent+=' '+next_word
        #print(next_word)
    
    return(final_sent)


def predict_next_word_add1(bigram):
    
    elements = []
    weights = []
    for i in generate_sent_map_trigram[bigram]:
        elements.append(i)
        weights.append(prob_add1_trigram[i+'|'+bigram])

    
    weights=np.array(weights)
    weights /= weights.sum()
    nextword = choice(elements, p = weights)
    return(nextword)

    
def generate_sent_add1():
    
    bigram = '<s> <s>'
    final_sent = '<s> '
    next_word = predict_next_word_add1(bigram)
    bigram = bigram.split()[1]+' '+next_word
    final_sent+=next_word
    while bigram.split()[1] != '</s>':
        next_word = predict_next_word_add1(bigram)
        bigram = bigram.split()[1]+' '+next_word
        final_sent+=' '+next_word
        #print(next_word)
    
    return(final_sent)


    
def get_unk_set(train_data):
    corpus=''
    unknown_set = set()
    
    for i in range(len(train_data)):
        corpus+=train_data[i]+' '
    
    a = sentence()
    a.createNgrams(corpus)
    
    del corpus
    unigrams_count = Counter(a.unigrams)
    
    for k,v in unigrams_count.items():
        if v < 1:
            unknown_set.add(k)
        
    return(unknown_set)
    
def replace_with_unk(train_data, unknown_set):    
    train_data_v2 = []

    for i in range(len(train_data)):
        k = train_data[i].split()
        sent = ''
        for j in range(len(k)):
            if k[j] in unknown_set:
                sent+='<UNK>'+' '
            else:
                sent+=k[j]+' '
        train_data_v2.append(sent)
        #print(train_data[i])
        #print(train_data_v2[i])
    return(train_data_v2)
    
    
def calc_perplexity_interpolation(testsent, l1, l2, l3):
    global prob_add1_trigram
    global prob_add1_bigram
    global prob_add1_unigram
    global trigrams_count
    global bigrams_count
    global vocab
    global prob_interpolation
    
    trigrams_list = get_trigrams(testsent)
    
    temp = 1
    for i in range(len(trigrams_list)):
        b = trigrams_list[i].split()
        key = b[2]+'|'+b[0]+' '+b[1]

        if key not in prob_add1_trigram:
            val1 = (0+1)/(bigrams_count[b[0]+' '+b[1]]+vocab)
        else:
            val1 = prob_add1_trigram[key]
	
        key = b[2]+'|'+b[1]
		
        if key not in prob_add1_bigram:
           val2 = (0+1)/(unigrams_count[b[1]]+vocab)
        else:
           val2 = prob_add1_bigram[key]
			
        key = b[2]
		
        if key not in prob_add1_unigram:
           val3 = (0+1)/(2*vocab)
        else:
           val3 = prob_add1_unigram[key]
        
        val = l1*val1+l2*val2+l3*val3
        prob_interpolation[b[2]+'|'+b[0]+' '+b[1]] = val

        temp8 = b[0]+' '+b[1]
        if temp8 in generate_sent_map_interpolation.keys():
            generate_sent_map_interpolation[temp8].add(b[2])
        else:
            generate_sent_map_interpolation[temp8] = set({b[2]})

        temp = temp / val

    perplexity = temp**(1/len(trigrams_list))
    return(perplexity)
    
    
def tot_perplexity(test_data_v2, l1, l2, l3):
    sum_perplexity=0
    for i in range(len(test_data_v2)):
        val = calc_perplexity_interpolation(test_data_v2[i], l1, l2, l3)
        sum_perplexity+=val
        #print(val)
    
    return([sum_perplexity/len(test_data_v2), l1, l2, l3])
    
    
def best_lambda(dev_data_v2):
    allperplexity=[]
       
    # Perplexity of dev data
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.1, 0.8))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.8, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.7, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.2, 0.7))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.6, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.3, 0.6))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.4, 0.5))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.1, 0.5, 0.4))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.1, 0.7))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.7, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.2, 0.6))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.6, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.3, 0.5))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.2, 0.5, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.1, 0.6))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.6, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.2, 0.5))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.5, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.3, 0.4))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.3, 0.4, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.4, 0.1, 0.5))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.4, 0.5, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.4, 0.2, 0.4))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.4, 0.4, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.4, 0.3, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.5, 0.1, 0.4))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.5, 0.4, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.5, 0.2, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.5, 0.3, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.6, 0.1, 0.3))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.6, 0.3, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.6, 0.2, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.7, 0.1, 0.2))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.7, 0.2, 0.1))
    allperplexity.append(tot_perplexity(dev_data_v2, 0.8, 0.1, 0.1))

    allperplexity.sort(key = operator.itemgetter(0), reverse = False)
    return(allperplexity[0])


    
###########################################################

# Main program
unigrams_count = dict()
bigrams_count = dict()
trigrams_count = dict()
prob_add1_trigram = dict()
prob_add1_bigram = dict()
prob_add1_unigram = dict()
prob_interpolation = dict()
generate_sent_map_trigram = dict()
generate_sent_map_bigram = dict()
generate_sent_map_interpolation = dict() 

print("Loading data..")


df= pd.read_csv("winemag-data_first150k.csv")
print("Pre-processing data..")
train_data = preprocess_data(df)
print("Replacing words with freq less than 5 as <UNK>..")
train_unk_set = get_unk_set(train_data)
train_data_v2 = replace_with_unk(train_data, train_unk_set)

corpus=''
for i in range(len(train_data_v2)):
    corpus+=train_data_v2[i]+' '

a = sentence()
a.createNgrams(corpus)

del corpus

unigrams_count = Counter(a.unigrams)
bigrams_count = Counter(a.bigrams)
trigrams_count = Counter(a.trigrams)
vocab = len(unigrams_count)
c = sum(unigrams_count.values())

print("Calculate add1 smoothing for unigrams, bigrams and trigrams..")
# Calc smoothing
for i in range(len(train_data_v2)):
    calc_add1_smoothing_trigram(train_data_v2[i])
    calc_add1_smoothing_bigram(train_data_v2[i])
    calc_add1_smoothing_unigram(train_data_v2[i])

#generate_sent()
#calc_perplexity_interpolation(train_data_v2, 0.1, 0.1, 0.9)
#################################################
# Test data
print("\n")
print("Running on test data..")

#df2 = pd.read_csv(base+'test_set.csv')
df2 = pd.read_csv('winemag-data_first150k.csv')
test_data = preprocess_data(df2)
test_unk_set = get_unk_set(test_data)

full_unk_set = train_unk_set.union(test_unk_set)
test_data_v2 = replace_with_unk(test_data, full_unk_set)


print("Calculating perplexity for add1 smoothing..")
# Calc perplexity
sum_perplexity=0
for i in range(len(test_data_v2)):
    val = calc_perplexity_add1(test_data_v2[i])
    sum_perplexity+=val
    
print("Perplexity of add1 smoothing: ", sum_perplexity/len(test_data_v2))
##################################################    
print("\n")
print("Calculating lambdas for simple interpolation on dev set..")

#df3 = pd.read_csv(base+'dev_set.csv')
df3 = pd.read_csv('winemag-data_first150k.csv')
dev_data = preprocess_data(df3)
dev_data_v2 = replace_with_unk(dev_data, full_unk_set)

best_result = best_lambda(dev_data_v2)
l1 = best_result[1]
l2 = best_result[2]
l3 = best_result[3]
print("Best lambda1: ", best_result[1])
print("Best lambda2: ", best_result[2])
print("Best lambda3: ", best_result[3])

print("\n")
print("Calculating perplexity for simple interpolation on test set..")


sum_perplexity=0
for i in range(len(test_data_v2)):
    val = calc_perplexity_interpolation(test_data_v2[i], l1, l2, l3)
    sum_perplexity+=val

print("Perplexity of interpolation smoothing: ", sum_perplexity/len(test_data_v2))
###################################################
print("\n")
print("Generating sentences using add1 smoothing:")

for i in range(20):
    fl=open("modeltrigram.dat", "a+")
    k = generate_sent_add1()
    print (str(i)+'.', k)
    text= k
    fl.write("%s\n" %text)
    fl.close()

print("\n")
print("Generating sentences using simple interpolation:")
print('Program executed')

   
   