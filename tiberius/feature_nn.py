#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: Neural Network Models that takes LSTM embeddings, Crude embedding and cosmetic features

"""

import pandas as pd
import numpy as np
import re
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers import TimeDistributed, Lambda
from keras.utils import np_utils
from keras.layers import Merge
from keras.layers import Lambda, SpatialDropout1D
from keras.preprocessing import text,sequence
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as KBE
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def train_test():

    #PICLKLE_FILE_PATH = "data/train/basic-nn-model.h5"
    SUBMISSION_FILE_PATH = "data/test/feature-nn-submission.csv"

    df = pd.read_csv("data/train/train.csv")
    test_df = pd.read_csv("data/test/test.csv")

    y = df.is_duplicate.values

    test_tknzr = text.Tokenizer(num_words=250000)
    tknzr = text.Tokenizer(num_words=250000)
    max_len = 40

    q1 = df.question1.values.astype(str)
    q2 = df.question2.values.astype(str)
    q1 = cleanser(list(q1))
    q2 = cleanser(list(q2))

    test_q1 = test_df.question1.values.astype(str)
    test_q2 = test_df.question2.values.astype(str)
    test_q1 = cleanser(list(test_q1))
    test_q2 = cleanser(list(test_q2))

    tknzr.fit_on_texts(list(q1)+list(q2))

    q1 = tknzr.texts_to_sequences(q1)
    q1 = sequence.pad_sequences(q1, maxlen=max_len)

    q2 = tknzr.texts_to_sequences(q2)
    q2 = sequence.pad_sequences(q2, maxlen=max_len)

    test_tknzr.fit_on_texts(list(test_q1) + list(test_q2))

    test_q1 = test_tknzr.texts_to_sequences(test_q1)
    test_q1 = sequence.pad_sequences(test_q1, maxlen=max_len)

    test_q2 = test_tknzr.texts_to_sequences(test_q2)
    test_q2 = sequence.pad_sequences(test_q2, maxlen=max_len)

    word_index = tknzr.word_index

    #ytrain_enc = np_utils.to_categorical(y)

    #------- LSTM ---------#

    model1 = Sequential()
    model1.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
    model1.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))

    model2 = Sequential()
    model2.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
    model2.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))

    #-------- Crude Embeddings --------#

    embeddings_index = {}
    f = open('data/vectors.txt')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model3 = Sequential()
    model3.add(Embedding(len(word_index) + 1,
                         300,
                         weights=[embedding_matrix],
                         input_length=40,
                         trainable=False))

    model3.add(TimeDistributed(Dense(300, activation='relu')))
    model3.add(Lambda(lambda x: KBE.sum(x, axis=1), output_shape=(300,)))

    model4 = Sequential()
    model4.add(Embedding(len(word_index) + 1,
                         300,
                         weights=[embedding_matrix],
                         input_length=40,
                         trainable=False))

    model4.add(TimeDistributed(Dense(300, activation='relu')))
    model4.add(Lambda(lambda x: KBE.sum(x, axis=1), output_shape=(300,)))

    # ---------- Feature Embedding ------------ #

    df = basic_features(df)
    test_df = basic_features(test_df)
    test_features = test_df[['len_q1', 'len_q2', 'diff_len',
           'len_char_q1', 'len_char_q2',
           'len_word_q1', 'len_word_q2',
           'common_words','q_ratio', 'w_ratio',
            'partial_ratio', 'partial_token_set_ratio',
            'partial_token_sort_ratio', 'token_set_ratio',
            'token_sort_ratio']]

    features = df[['len_q1', 'len_q2', 'diff_len',
           'len_char_q1', 'len_char_q2',
           'len_word_q1', 'len_word_q2',
           'common_words','q_ratio', 'w_ratio',
            'partial_ratio', 'partial_token_set_ratio',
            'partial_token_sort_ratio', 'token_set_ratio',
            'token_sort_ratio']]

    SS = StandardScaler()
    SS.fit(np.vstack((features, test_features)))
    features = SS.transform(features)
    test_features = SS.transform(test_features)

    model5 = Sequential()
    model5.add(Dense(16,input_shape=features.shape[1],activation='relu'))

    merged_model = Sequential()
    merged_model.add(Merge([model1, model2, model3, model4, model5], mode='concat'))
    merged_model.add(BatchNormalization())

    merged_model.add(Dense(300))
    merged_model.add(PReLU())
    merged_model.add(Dropout(0.2))
    merged_model.add(BatchNormalization())

    merged_model.add(Dense(300))
    merged_model.add(PReLU())
    merged_model.add(Dropout(0.2))
    merged_model.add(BatchNormalization())

    merged_model.add(Dense(300))
    merged_model.add(PReLU())
    merged_model.add(Dropout(0.2))
    merged_model.add(BatchNormalization())

    merged_model.add(Dense(1))
    merged_model.add(Activation('sigmoid'))

    merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

    merged_model.fit([q1, q2], y=y, batch_size=384, epochs=200,verbose=1)#, validation_split=0.25, shuffle=True)#, callbacks=[checkpoint])

    result = merged_model.predict_proba([test_q1, test_q2, test_q1, test_q2, test_features], batch_size=384, verbose=1)

    with open(SUBMISSION_FILE_PATH, 'w') as submission_file:
        submission_file.write('test_id,is_duplicate' + '\n')

        for i in range(0, len(result)):
            submission_file.write(str(i) + ',' + str('%.1f' % result[i][0]) + '\n')


def basic_features(data):

    data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
    data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
    data['diff_len'] = data.len_q1 - data.len_q2
    data['len_char_q1'] = data.question1.apply(lambda x: len("".join(str(x).split())))
    data['len_char_q2'] = data.question2.apply(lambda x: len("".join(str(x).split())))
    data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
    data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
    data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).
                                                    intersection(set(str(x['question2']).lower().split()))), axis=1)

    data['q_ratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['w_ratio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['partial_token_set_ratio'] = data.apply(
        lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])),
        axis=1)
    data['partial_token_sort_ratio'] = data.apply(
        lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])),
        axis=1)
    data['token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])),
                                         axis=1)
    data['token_sort_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])),
                                          axis=1)

    return data

def cleanser(questions):

    for i,text in enumerate(questions):
        text = text.lower().split()
        # Optionally, remove stop words
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

        text = " ".join(text)

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        # Optionally, shorten words to their stems
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        questions[i] = " ".join(stemmed_words)

    return questions

def main():
    train_test()

main()