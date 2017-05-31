from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Lambda,Merge,TimeDistributed,BatchNormalization,PReLU,Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing import sequence,text
from keras.layers.core import Activation,Dense,SpatialDropout1D
import logging
import multiprocessing
import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras import backend as K
import tensorflow as tf
import sys
from keras import optimizers


model_file='wiki.en.vec'
MAX_LEN=40
MAX_NB_WORDS=200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM=300


def Angle(inputs):

    length_input_1 = K.sqrt(K.sum(tf.pow(inputs[0], 2), axis=1, keepdims=True))
    length_input_2 = K.sqrt(K.sum(tf.pow(inputs[1], 2), axis=1, keepdims=True))
    result = K.batch_dot(inputs[0], inputs[1], axes=1) / (length_input_1 * length_input_2)
    angle = tf.acos(result)
    return angle


def Distance(inputs):

    s = inputs[0] - inputs[1]
    output = K.sum(s ** 2, axis=1, keepdims=True)
    return output

def deep_net(train,test):



    y=train.is_duplicate.values
    tk=text.Tokenizer(MAX_NB_WORDS)
    tk.fit_on_texts(list(train.question1.values.astype(str))+list(train.question2.values.astype(str)))
    question1 = tk.texts_to_sequences(train.question1.values.astype(str))
    question1 = sequence.pad_sequences(question1,maxlen=MAX_LEN)
    question2 = tk.texts_to_sequences(train.question2.values.astype(str))
    question2 = sequence.pad_sequences(question2,maxlen=MAX_LEN)

    tk = text.Tokenizer()
    test_ids = test.test_id.values
    tk.fit_on_texts(list(test.question1.values.astype(str)) + list(test.question2.values.astype(str)))
    test_question1 = tk.texts_to_sequences(test.question1.values.astype(str))
    test_question1 = sequence.pad_sequences(test_question1, maxlen=MAX_LEN)
    test_question2 = tk.texts_to_sequences(test.question2.values.astype(str))
    test_question2 = sequence.pad_sequences(test_question2, maxlen=MAX_LEN)

    word_index = tk.word_index
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

        #--------question1--------#

    model1 = Sequential()
    print "Build Model"

    model1.add(Embedding(
        len(word_index)+1,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=False
        ))

    model1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
    model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM,)))



    print model1.summary()

  #---------question2-------#

    model2=Sequential()

    model2.add(Embedding(
        len(word_index) + 1,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=False
        ))


    model2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
    model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM,)))


    print model2.summary()

    # ---------Merged------#

    # Here you get question embedding

    # Calculate distance between vectors
    Distance_merged_model = Sequential()
    Distance_merged_model.add(Merge(layers=[model1, model2], mode=Distance, output_shape=(1,)))

    print Distance_merged_model.summary()

    # Calculate Angle between vectors

    Angle_merged_model = Sequential()
    Angle_merged_model.add(Merge(layers=[model1, model2], mode=Angle, output_shape=(1,)))

    print Angle_merged_model.summary()

    neural_network = Sequential()
    neural_network.add(Merge(layers=[Distance_merged_model, Angle_merged_model], mode='concat'))
    neural_network.add(Dense(2, input_shape=(1,)))
    neural_network.add(Dense(1))

    neural_network.add(Activation('sigmoid'))

    print neural_network.summary()

    sgd = optimizers.SGD(lr=0.01, clipnorm=0.5)
    neural_network.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint('LSTM_DISTANCE_WEIGHTS.h5', monitor='val_acc', save_best_only=True, verbose=2)

    hist=neural_network.fit([question1,question2],y=y, batch_size=100, epochs=150,shuffle=True,
                    verbose=1,validation_split=0.1,callbacks=[early_stopping,checkpoint])

    neural_network.load_weights("LSTM_DISTANCE_WEIGHTS.h5")
    bst_val_score = min(hist.history['val_loss'])
    bst_val_acc=max(hist.history['val_acc'])
    print bst_val_acc

    result=neural_network.predict_proba([test_question1,test_question2], batch_size=100, verbose=1)

    submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': result.ravel()})
    submission.to_csv('%.4f_' % (bst_val_score) + "submission" + '.csv', index=False)






def Main():

    test=pd.read_csv('test.csv')
    train = pd.read_csv('train.csv')
    deep_net(train,test)


Main()