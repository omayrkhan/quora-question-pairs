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
from keras.layers import Lambda, SpatialDropout1D
from keras.preprocessing import text,sequence
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
#import h5py

PICLKLE_FILE_PATH = "data/train/basic-nn-model.h5"
SUBMISSION_FILE_PATH = "data/test/basic-nn-model.csv"

df = pd.read_csv("data/train/train.csv")
y = df.is_duplicate.values

tknzr = text.Tokenizer(num_words=250000)
max_len = 100

tknzr.fit_on_texts(list(df.question1.values.astype(str))+list(df.question2.values.astype(str)))
q1 = tknzr.texts_to_sequences(df.question1.values)
q1 = sequence.pad_sequences(q1, maxlen=max_len)

q2 = tknzr.texts_to_sequences(df.question2.values.astype(str))
q2 = sequence.pad_sequences(q2, maxlen=max_len)

word_index = tknzr.word_index

ytrain_enc = np_utils.to_categorical(y)

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1, 300, input_length=100, dropout=0.2))
model1.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1, 300, input_length=100, dropout=0.2))
model2.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))

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

merged_model.fit([q1, q2], y=y, batch_size=384, epochs=2s00,verbose=1)
                    #, validation_split=0.25, shuffle=True, callbacks=[checkpoint])

#merged_model.save(PICLKLE_FILE_PATH)

output_df = pd.read_csv("data/test/test.csv")

output_tknzr = text.Tokenizer(nb_words=250000)

output_tknzr.fit_on_texts(list(output_df.question1.values.astype(str))+list(output_df.question2.values.astype(str)))
output_q1 = tknzr.texts_to_sequences(output_df.question1.values)
output_q1 = sequence.pad_sequences(output_q1, maxlen=max_len)

output_q2 = tknzr.texts_to_sequences(output_df.question2.values.astype(str))
output_q2 = sequence.pad_sequences(output_q2, maxlen=max_len)

result = merged_model.predict_proba([output_q1,output_q2], batch_size=384, verbose=1)

with open(SUBMISSION_FILE_PATH, 'w') as submission_file:
    submission_file.write('test_id,is_duplicate' + '\n')

    for i in range(0, len(result)):
        submission_file.write(str(i) + ',' + str('%.1f' % result[i][0]) + '\n')