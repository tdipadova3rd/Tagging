# Tony DiPadova and Steven Jiang, May 2018
# Written in Python 3
try:
	print("Checking dependencies...")
	from gensim.models import KeyedVectors
	from nltk.corpus.reader.conll import ConllCorpusReader
	import nltk
	from nltk import conlltags2tree
	from langdetect import detect
	import re
	from math import floor
	import numpy as np
	import sys
	import pandas as pd
	import itertools
	import matplotlib.pyplot as plt
	from keras.layers.recurrent import LSTM
	from keras.models import Sequential, Model, load_model
	from keras.layers import Dense, Bidirectional, Flatten, Dropout, TimeDistributed
	from keras.wrappers.scikit_learn import KerasClassifier
	from keras.layers.normalization import BatchNormalization
	from keras.utils import to_categorical
	import keras.backend as K
	from keras_contrib.layers import CRF
	from keras_contrib.utils import save_load_utils
	from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
	from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
except Exception as e:
	raise Exception("You do not have the necessary dependencies install. Try 'pip install -r requirements.txt' first.\n", e)

# global variables
max_length = 70
num_features = 300

classes = ['B-MISC', 'I-MISC', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'O']
num_classes = len(classes)
encoded_classes = range(num_classes)
class2idx = {classes[enc]: enc for enc in encoded_classes}

dropout = 0.1
recurrent_dropout = 0.3
hidden_nodes = 100
window_size = 70

nums_regex = re.compile(r'0+')


def clean_sents(sents, max_length):
    cleaned = []
    # remove sentences shorter than 5 words
    for sent in sents:
        if len(sent) > 4 and len(sent) <= max_length:
            new_sent = []
            # clean the words
            for word in sent:
                this_word = word.lower()
                new_word = ''
                # replace numbers with 0
                for char in this_word:
                    if char.isalpha():
                        new_word = new_word + char
                    elif char.isdigit():
                        new_word = new_word + '0'
                new_word = nums_regex.sub('0', new_word)
                new_sent.append(this_word)
            cleaned.append(new_sent)
    return cleaned

def arr2label(cats, labels):
    new_labels = []
    for i in range(len(cats)):
        sent_labels = []
        for j in range(len(cats[i])):
            label = np.argmax(cats[i][j])
            label = labels[label]
            new_labels.append(label)
    return new_labels

def tag2tree(tags):
    string = '(S'
    for i in range(len(tags)):
        # beginning case
        if i == 0 and tags[i][1] != 'O':
            string += ' (' + tags[i][1][2:] + ' ' + tags[i][0]  # beginning is B-
            if i + 1 < len(tags) and tags[i+1][1] == 'O':
                string += ')'
        elif tags[i][1] == 'O':
            string += ' ' + tags[i][0]
        else:
            # middle cases
            if i < len(tags) - 1:
                if tags[i+1][1] == 'O' and tags[i][1][0] == 'I':
                    string += ' ' + tags[i][0] + ')'
                elif tags[i+1][1][0] == 'I' and tags[i][1][0] == 'I':
                    string += ' ' + tags[i][0] 
                elif tags[i][1][0] == 'B' and tags[i+1][1] == 'O':
                    string += ' (' + tags[i][1][2:] + ' ' + tags[i][0] + ')'
                elif tags[i][1][0] == 'B' and tags[i+1][1][0] == 'I':
                    string += ' (' + tags[i][1][2:] + ' ' + tags[i][0]
            else:
                # end cases
                if tags[i][1][0] == 'B':
                    string += ' (' + tags[i][1][2:] + ' ' + tags[i][0] + ')'
                elif tags[i][1][0] == 'I':
                    string += ' ' + tags[i][0] + ')'
                
    string += ')'
    tree = nltk.Tree.fromstring(string)
    return tree

def get_padded_sentence_features(sentences, num_features, max_length, wv):
    features = np.empty((0, max_length, num_features))
    for i in range(len(sentences)):
        sent = sentences[i]
        new_sent = []
        for j in range(max_length):
            if 0 <= j < len(sent):
                this_word = sent[j]
                if this_word in wv.vocab:
                    new_sent.append(wv.get_vector(this_word))
                elif this_word == '':
                    new_sent.append(np.zeros(num_features))
                else:
                    new_sent.append(np.random.uniform(-0.25,0.25, num_features))  # random vector for unknown
            else:
                new_sent.append(np.zeros(num_features))

        feature_stack = np.dstack([[new_sent]])
        features = np.vstack([features, feature_stack])
        
    return features


def create_model(num_classes, num_features, hidden_nodes=100):
    model = Sequential()
    model.add(Bidirectional(
        LSTM(units=num_features, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout),
        input_shape=(window_size, num_features,),
        merge_mode='concat'))
    model.add(TimeDistributed(Dense(hidden_nodes, activation='relu')))
    # add a CRF layer to enforce NER IOB rules
    crf = CRF(num_classes, sparse_target=False)
    model.add(crf)
    model.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
    return model

def load_resources():
	print("Loading embeddings...")
	embeddings = {
		'en': KeyedVectors.load_word2vec_format('data/wiki.multi.en.vec.txt', binary=False),
		'es': KeyedVectors.load_word2vec_format('data/wiki.multi.es.vec.txt', binary=False),
		'de': KeyedVectors.load_word2vec_format('data/wiki.multi.de.vec.txt', binary=False),
		'it': KeyedVectors.load_word2vec_format('data/wiki.multi.it.vec.txt', binary=False)
	}

	print("Loading model...")
	model = create_model(num_classes, num_features)
	save_load_utils.load_all_weights(model,'models/full_train.h5', include_optimizer=False)

	return embeddings, model

def parse(sentence, model, wordvecs):
	sent = nltk.word_tokenize(sentence)  # tokenize the sentence
	if len(sent) > max_length:
		print("Sentence must be less than", max_length, "words long.")
		return tag2tree([('', 'O')])
	elif len(sent) <= 4:
		print("Sentence must be at least 5 words long.")
		return tag2tree([('', 'O')])
	cleaned = clean_sents([sent], max_length)  # normalize the sentence
	X = get_padded_sentence_features(cleaned, num_features, max_length, wordvecs)  # process the sentence

	pred = model.predict(X)  # predict
	labels = arr2label(pred, classes)  # get labels
	labels = labels[:len(sent)]  # remove padding
	
	tags = [(sent[i], labels[i]) for i in range(len(sent))]  # get the tags
	tree = tag2tree(tags)  # convert to tree

	return tree

if __name__ == '__main__':
	print("Loading resources, this may take several minutes...")
	embeddings, model = load_resources()
	print("Type any sentence in English, German, Spanish, or Italian to parse it into a Named Entity Tree.")
	print("Type 'q' to quit.")
	ipt = input('Sentence: ')
	while ipt != 'q':
		try:
			language = detect(ipt)
		except Exception as e:
			print("Unable to detect language, defaulting to English")
			language = 'en'
		if language in embeddings:
			tree = parse(ipt, model, embeddings[language])
			tree.pprint()
			print()
		else:
			print("Detected language:", language)
			print("Defaulting to English")
			tree = parse(ipt, model, embeddings['en'])
			tree.pprint()
			print()
		ipt = input('Sentence: ')



