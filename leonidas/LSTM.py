from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Lambda,Merge,TimeDistributed,BatchNormalization,PReLU
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
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


#----Global Variables-----#

MAX_LEN = 40



def Angle(inputs):

     length_input_1=K.sqrt(K.sum(tf.pow(inputs[0],2),axis=1,keepdims=True))
     length_input_2=K.sqrt(K.sum(tf.pow(inputs[1],2),axis=1,keepdims=True))
     result=K.batch_dot(inputs[0],inputs[1],axes=1)/(length_input_1*length_input_2)
     angle = tf.acos(result)
     return angle

def Distance(inputs):

    s = inputs[0] - inputs[1]
    output = K.sum(s ** 2,axis=1,keepdims=True)
    return output


def Data_Cleaning(corpus_raw):

    corpus_raw['question1'] = corpus_raw['question1'].apply(lambda row: str(row).lower())
    corpus_raw['question2'] = corpus_raw['question2'].apply(lambda row: str(row).lower())



    # preserve contractions and ingore punctuations occuring in sentences--- remove punctuation between words except "'"
    corpus_raw['question1'] = corpus_raw['question1'].apply(lambda row: "".join([s if s.isalpha()  or s.isdigit() or s=="'" or s==" " else ' ' for s in row ]))
    corpus_raw['question2'] = corpus_raw['question2'].apply(lambda row: "".join([s  if s.isalpha() or s.isdigit() or s == "'" or s == " " else ' ' for s in row]))




    return corpus_raw


def deep_net(train,test):

    data=Data_Cleaning(train)


    y=data.is_duplicate.values
    tk=text.Tokenizer()
    tk.fit_on_texts(list(data.question1.values)+list(data.question2.values))


    question1 = tk.texts_to_sequences(data.question1.values)
    question1 = sequence.pad_sequences(question1,maxlen=MAX_LEN)

    question2 = tk.texts_to_sequences(data.question2.values)
    question2 = sequence.pad_sequences(question2,maxlen=MAX_LEN)


    word_index = tk.word_index
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    num_features = 300
    num_workers = multiprocessing.cpu_count()
    context_size = 5
    downsampling = 7.5e-06
    seed = 1
    min_word_count = 5
    hs = 1
    negative = 5


    Quora_word2vec = gensim.models.Word2Vec(

        sg=0,
        seed=1,
        workers=num_workers,
        min_count=min_word_count,
        size=num_features,
        window=context_size,  # (2 and 5)
        hs=hs,  # (1 and 0)
        negative=negative,  # (5 and 10)
        sample=downsampling  # (range (0, 1e-5). )

    )

    Quora_word2vec = gensim.models.KeyedVectors.load_word2vec_format('wiki.en.vec',binary=False)
    embedding_matrix=np.zeros((len(word_index)+1,300))

    for word , i in tqdm(word_index.items()): #i is index

        try:

            embedding_vector =  Quora_word2vec[word] #Exception is thrown if there is key error
            embedding_matrix[i] = embedding_vector

        except Exception as e:  #If word is not found continue

            continue
#--------question1--------#

    model1 = Sequential()
    print "Build Model"

    model1.add(Embedding(
        len(word_index)+1,
        300,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=False
        ))


    model1.add(TimeDistributed(Dense(300, activation='relu')))
    model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))
    #model1.add(SpatialDropout1D(0.2))


    print model1.summary()

  #---------question2-------#

    model2=Sequential()

    model2.add(Embedding(
        len(word_index) + 1,
        300,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=False
        ))  # Embedding layer

    model2.add(TimeDistributed(Dense(300, activation='relu')))
    model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))
    #model2.add(SpatialDropout1D(0.2))

    print model2.summary()


    #---------Merged------#

    #Here you get question embedding

    #Calculate distance between vectors
    Distance_merged_model=Sequential()
    Distance_merged_model.add(Merge(layers=[model1, model2], mode=Distance, output_shape=(1,)))

    print Distance_merged_model.summary()

    #Calculate Angle between vectors

    Angle_merged_model=Sequential()
    Angle_merged_model.add(Merge(layers=[model1,model2],mode=Angle,output_shape=(1,)))

    print Angle_merged_model.summary()


    neural_network = Sequential()
    neural_network.add(Merge(layers=[Distance_merged_model,Angle_merged_model],mode='concat'))
    neural_network.add(Dense(2,input_shape=(1,)))
    neural_network.add(Dense(1))




    neural_network.add(Activation('sigmoid'))

    print neural_network.summary()


    neural_network.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    #checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

    #history_callback=neural_network.fit([question1,question2],y=y,batch_size=384,validation_split=0.3,verbose=1,
                       #shuffle=True,callbacks=[checkpoint],nb_epoch=200)

    #train_loss=history_callback.history["loss"]
    #val_loss = history_callback.history["val_loss"]
    #train_acc=history_callback['acc']
    #val_acc=history_callback['val_acc']
    #np.savetxt('train_loss.txt',train_loss)
    #np.savetxt('val_loss.txt',val_loss)
    #np.savetxt('train_acc.txt',train_acc)
    #np.savetxt('val_acc.txt',val_acc)


    neural_network.fit([question1,question2],y=y, batch_size=384, epochs=200,
                     verbose=1)



    data_test=Data_Cleaning(test)
    tk = text.Tokenizer()
    tk.fit_on_texts(list(data_test.question1.values) + list(data_test.question2.values))

    test_question1 = tk.texts_to_sequences(data.question1.values)
    test_question1 = sequence.pad_sequences(test_question1, maxlen=MAX_LEN)

    test_question2 = tk.texts_to_sequences(data.question2.values)
    test_question2 = sequence.pad_sequences(test_question2, maxlen=MAX_LEN)
    result=neural_network.predict_proba([test_question1,test_question2], batch_size=384, verbose=1)

    with open("LSTM_SUBMISSION_FILE.csv", 'w') as submission_file:
        submission_file.write('test_id,is_duplicate' + '\n')

        for i in range(0, len(result)):
            submission_file.write(str(i) + ',' + str('%.1f' % result[i][0]) + '\n')





def Main():

    test=pd.read_csv(sys.argv[2],nrows=1000)
    train = pd.read_csv(sys.argv[1])
    deep_net(train,test)



Main()


