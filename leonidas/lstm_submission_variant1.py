from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Lambda,Merge,TimeDistributed,BatchNormalization,PReLU,Dropout,Input,Convolution1D,MaxPooling1D,GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing import sequence,text
from keras.layers.core import Activation,Dense,SpatialDropout1D
from keras.models import Model
from keras.layers.merge import concatenate
from tqdm import tqdm

import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from string import punctuation
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import StandardScaler
from fuzzywuzzy import fuzz

num_lstm = 228
num_dense = 114
rate_drop_lstm = 0.17
rate_drop_dense = 0.25
EMBEDDING_DIM=300
MAX_LEN=30
filter_length = 5
nb_filter = 64
pool_length = 4

#LEAKY FEATURES

def Leaky_Features():

    df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    #-----LEAKY FEATURES--------------------------------------#


    ques = pd.concat([df[['question1', 'question2']],
                      test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')

    q_dict = defaultdict(set)
    for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

    df['q1_q2_intersect'] = df.apply(lambda row :len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))) ,
                                                                axis=1, raw=True)
    df['q1_freq'] = df.apply(lambda row : len(q_dict[row['question1']]), axis=1, raw=True)
    df['q2_freq'] = df.apply(lambda row :len(q_dict[row['question2']]) , axis=1, raw=True)

    test_df['q1_q2_intersect'] = test_df.apply(lambda row : len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']])))
                                           , axis=1, raw=True)
    test_df['q1_freq'] = test_df.apply(lambda row: len(q_dict[row['question2']]), axis=1, raw=True)
    test_df['q2_freq'] = test_df.apply(lambda row :len(q_dict[row['question1']]) , axis=1, raw=True)

    train_leaks = df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
    test_leaks = test_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]

    ss = StandardScaler()
    ss.fit(np.vstack((train_leaks, test_leaks)))
    train_leaks = ss.transform(train_leaks)
    test_leaks = ss.transform(test_leaks)

    train_leaks=np.vstack((train_leaks,train_leaks))

    return train_leaks,test_leaks


#------------------------------------------------------------#

def train_test():

    SUBMISSION_FILE_PATH = "feature-nn-submission.csv"
    train_leaks, test_leaks = Leaky_Features()

    df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")



    y = np.array(df.is_duplicate.values)

    max_len = 30

    q1 = df.question1.values.astype(str)
    q2 = df.question2.values.astype(str)
    q1 = cleanser(list(q1))
    q2 = cleanser(list(q2))

    test_q1 = test_df.question1.values.astype(str)
    test_q2 = test_df.question2.values.astype(str)
    test_q1 = cleanser(list(test_q1))
    test_q2 = cleanser(list(test_q2))

    tknzr = text.Tokenizer(num_words=200000)
    tknzr.fit_on_texts(q1 + q2 + test_q1 + test_q2)

    q1 = tknzr.texts_to_sequences(q1)
    q1 = sequence.pad_sequences(q1, maxlen = max_len)

    q2 = tknzr.texts_to_sequences(q2)
    q2 = sequence.pad_sequences(q2, maxlen = max_len)

    test_q1 = tknzr.texts_to_sequences(test_q1)
    test_q1 = sequence.pad_sequences(test_q1, maxlen=max_len)

    test_q2 = tknzr.texts_to_sequences(test_q2)
    test_q2 = sequence.pad_sequences(test_q2, maxlen=max_len)

    word_index = tknzr.word_index
    size_words = min(200000, len(word_index)) + 1

    test_q1 = np.array(test_q1)
    test_q2 = np.array(test_q2)

    extended_q1 = np.vstack((q1, q2))
    extended_q2 = np.vstack((q2, q1))
    extended_y = np.vstack((y.reshape(len(df),1), y.reshape(len(df),1)))
    extended_y = extended_y.reshape(2*len(df),)

    print "text preprocessing and tokenier initialization done"


    #-------- GLoVe Initialization --------#

    print "GloVe initalization starts"
    embeddings_index = {}
    f = open('vectors.txt')
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

    # ---------- Feature Embedding ------------ #

    temp_df = df[['question1', 'question2']]
    temp_df.columns = ['question2', 'question1']

    extended_df = pd.concat([df[['question1', 'question2']],
                                 temp_df[['question1', 'question2']]], axis=0).reset_index(drop='index')

    extended_df = basic_features(extended_df)
    test_df = basic_features(test_df)
    test_features = test_df[['len_q1', 'len_q2', 'diff_len',
                                 'len_char_q1', 'len_char_q2',
                                 'len_word_q1', 'len_word_q2',
                                 'common_words', 'q_ratio', 'w_ratio',
                                 'partial_ratio', 'partial_token_set_ratio',
                                 'partial_token_sort_ratio', 'token_set_ratio',
                                 'token_sort_ratio']]

    features = extended_df[['len_q1', 'len_q2', 'diff_len',
                                'len_char_q1', 'len_char_q2',
                                'len_word_q1', 'len_word_q2',
                                'common_words', 'q_ratio', 'w_ratio',
                                'partial_ratio', 'partial_token_set_ratio',
                                'partial_token_sort_ratio', 'token_set_ratio',
                                'token_sort_ratio']]

    features = np.array(features)
    test_features = np.array(test_features)
    #features = np.vstack((features,features))

    print "basic features done"
    SS = StandardScaler()
    SS.fit(np.vstack((features, test_features)))
    features = SS.transform(features)
    test_features = SS.transform(test_features)

    # -------- MODEL--------#

    model1 = Sequential()

    model1.add(Embedding(size_words, 300, weights=[embedding_matrix], input_length=30, trainable=False))
    model1.add(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    model2 = Sequential()

    model2.add(Embedding(size_words, 300, weights=[embedding_matrix], input_length=30, trainable=False))
    model2.add(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    model5 = Sequential()
    model5.add(Dense(num_dense,input_shape=(features.shape[1],),activation='relu'))

    model3 = Sequential()
    model3.add(Dense(num_dense / 2, input_shape=(train_leaks.shape[1],), activation='relu'))

    model4 = Sequential()

    model4.add(Embedding(len(word_index) + 1,
                         EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=MAX_LEN,
                         trainable=False))

    model4.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1
                             ))
    model4.add(Dropout(rate_drop_dense))
    model4.add(MaxPooling1D(pool_size=pool_length))
    model4.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1
                             ))
    model4.add(Dropout(rate_drop_dense))
    model4.add(GlobalMaxPooling1D())

    model6 = Sequential()
    model6.add(Embedding(len(word_index) + 1,
                         EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=MAX_LEN,
                         trainable=False))

    model6.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1
                             ))
    model6.add(Dropout(rate_drop_dense))
    model6.add(MaxPooling1D(pool_size=pool_length))
    model6.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1
                             ))
    model6.add(Dropout(rate_drop_dense))
    model6.add(GlobalMaxPooling1D())

    merged_model = Sequential()
    merged_model.add(Merge([model1, model2,model5, model3,model4,model6], mode='concat'))
    merged_model.add(BatchNormalization())
    merged_model.add(Dropout(rate_drop_dense))

    merged_model.add(Dense(num_dense,activation='relu'))
    merged_model.add(BatchNormalization())
    merged_model.add(Dropout(rate_drop_dense))

    merged_model.add(Dense(num_dense, activation='relu'))
    merged_model.add(BatchNormalization())
    merged_model.add(Dropout(rate_drop_dense))

    merged_model.add(Dense(num_dense, activation='relu'))
    merged_model.add(BatchNormalization())
    merged_model.add(Dropout(rate_drop_dense))

    merged_model.add(Dense(num_dense, activation='relu'))
    merged_model.add(BatchNormalization())
    merged_model.add(Dropout(rate_drop_dense))

    merged_model.add(Dense(1))
    merged_model.add(Activation('sigmoid'))

    merged_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    checkpoint = ModelCheckpoint('variant1-feature-nn-weights.h5', monitor='val_loss', save_best_only=True, verbose=2)



    hist = merged_model.fit([extended_q1, extended_q2, features,train_leaks,extended_q1,extended_q2],
                            y=extended_y,
                     batch_size=384, epochs=200, verbose=1, validation_split=0.1,
                     shuffle=True, callbacks=[early_stopping,checkpoint])

    bst_val_score = min(hist.history['val_loss'])
    merged_model.load_weights('variant1-feature-nn-weights.h5')

    result = merged_model.predict_proba([test_q1, test_q2, test_features,test_leaks,test_q1,test_q2], batch_size=384, verbose=1)
    result += merged_model.predict_proba([test_q2, test_q1, test_features,test_leaks,test_q1,test_q2], batch_size=384, verbose=1)
    result /= 2

    submission = pd.DataFrame({'test_id': np.array(test_df.test_id.values), 'is_duplicate': result.ravel()})
    submission.to_csv('%.4f_' % (bst_val_score) + 'lstm-submission_variant1.csv', index=False)



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

        text = text.lower()

        text = ("".join([s if s.isalpha() or s.isdigit()
                             or s == "'" or s == " " else ' ' for s in text]))

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


        questions[i] = text

    return questions


def main():
    train_test()

main()