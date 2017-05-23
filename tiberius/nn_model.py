#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: Neural Network Models for Quora Questions

"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.utils import np_utils
from keras.layers import Merge
from keras.layers import Lambda
from keras.preprocessing import text,sequence
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import time

df = pd.read_csv("data/sample/sample.csv")
y = df.is_duplicate.values

tknzr = text.Tokenizer(nb_words=250000)
max_len = 100

tknzr.fit_on_texts(list(df.question1.values.astype(str))+list(df.question2.values.astype(str)))
x1 = tknzr.texts_to_sequences(df.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tknzr.texts_to_sequences(df.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

word_index = tknzr.word_index

ytrain_enc = np_utils.to_categorical(y)

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1, 300, input_length=100, dropout=0.2))
model1.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1, 300, input_length=100, dropout=0.2))
model2.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

merged_model = Sequential()
merged_model.add(Merge([model1, model2], mode='concat'))
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

merged_model.fit([x1, x2], y=y, batch_size=384, nb_epoch=10,
                 verbose=1, validation_split=0.1, shuffle=True)#, callbacks=[checkpoint])
