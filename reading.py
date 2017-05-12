#!/usr/local/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: quora feature engineering and preprocessing

"""

import pandas as pd
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
import numpy as np
import time
import gensim
from scipy.spatial.distance import cosine, cityblock, canberra, minkowski, braycurtis, euclidean, jaccard
from scipy.stats import skew, kurtosis
import string
import enchant
from num2words import num2words
import glove

MODEL_FILE_PATH = "data/wiki.en.vec"
MISSING_WORD_FILE_PATH = "data/wiki/missing_words_test.csv"
TRANSFORMED_DATA_FILE_PATH = "data/wiki/new_test.csv"
def question_to_vector(column1,column2):

    model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_FILE_PATH, binary=False)
    print "model loaded"

    combined_text = zip(column1,column2)
    temp_container = np.zeros(shape=(len(column1),615))
    missing_words_df = pd.DataFrame(columns=['id','question1','question2'])

    for i, questions in enumerate(combined_text):

        question_vectors, missing_words, zero_flag = vectorizer(questions,model)
        temp_container[i, 0:300] = question_vectors[0, :]
        temp_container[i, 300:600] = question_vectors[1, :]

        distance_dict = distances(question_vectors[0, :], question_vectors[1, :])
        for j,key in enumerate(distance_dict.keys()):
            temp_container[i, 602+j] = distance_dict[key]

        temp_container[i, 613] = zero_flag[0]
        temp_container[i, 614] = zero_flag[1]

        if len(missing_words[0]) > 0 or len(missing_words[0]) > 1:
            missing_words_df.loc[i] = [int(i),','.join(map(str, missing_words[0])),','.join(map(str, missing_words[1]))]

    WMD_temp = word_movers_distance(combined_text,model)
    temp_container[:, 600] = WMD_temp[:,0]
    temp_container[:, 601] = WMD_temp[:,1]
    missing_words_df.to_csv(MISSING_WORD_FILE_PATH,sep='\t')
    #print "missing words written"

    return temp_container,distance_dict.keys()


def cleanser(sentence):

    sentence = str(sentence).lower()
    sentence = ''.join(s if s.isalpha() or s.isdigit() or s == "'" or s == " " else " " for s in sentence)
    words = str(sentence).split()
    words = [word for word in words if word not in stopwords.words('english')]
    words = [word for word in words if word.isalpha()]

    return words


def vectorizer(questions, model):

    questions = [questions[0],questions[1]]
    M, temp_words, missing_words, question_vectors = [],[],[],np.zeros(shape=(2,300))
    zero_flag, token_list = [], []
    i = 0

    for question in questions:
        try:
            words = cleanser(question)
            if len(words) == 0:
                M.append(np.zeros(shape=(300,))+np.nan)
            else:
                for word in words:
                    try:
                        M.append(model[word])
                    except Exception as e:
                        #print "*** inner exception ***  "+str(e)
                        temp_words.append(word)

            if len(M) == 0:
                M.append(np.zeros(shape=(300,)) + np.nan)
            M = np.array(M)
            v = M.sum(axis=0)
            v = v / np.sqrt((v ** 2).sum())

        except Exception as e:
            #print "*** outer exception ***"+str(e)
            v = np.zeros(shape=(300,))+np.nan

        if np.count_nonzero(~np.isnan(v)) > 0:
            zero_flag.append(0)
        else:
            zero_flag.append(1)
        question_vectors[i,:] = v.reshape(1,300)
        i += 1
        missing_words.append(temp_words)
        temp_words, M, v = [],[],0

    return question_vectors, missing_words, zero_flag[::-1]


def word_movers_distance(questions, model):

    WMD = np.zeros(shape=[len(questions),2])
    for i, question in enumerate(questions):
        try:
            WMD[i,0] = model.wmdistance(cleanser(question[0]), cleanser(question[1]))
        except Exception as e:
            print "*** WMD_basic exception ***  " + str(e)
            WMD[i, 0] = np.nan

    model.init_sims(replace=True)
    for i, question in enumerate(questions):
        try:
            WMD[i, 1] = model.wmdistance(cleanser(question[0]), cleanser(question[1]))
        except Exception as e:
            print "*** WMD_normalised exception ***  " + str(e)
            WMD[i, 1] = np.nan

    return WMD


def distances(vectorA, vectorB):

    distance_dict = {}

    try:
        distance_dict['cosine'] = cosine(vectorA, vectorB)
    except Exception as e:
        #print "*** cosine exception ***  " + str(e)
        distance_dict['cosine'] = np.nan

    try:
        distance_dict['cityblock'] = cityblock(vectorA, vectorB)
    except Exception as e:
        #print "*** cityblock exception ***  " + str(e)
        distance_dict['cityblock'] = np.nan

    try:
        distance_dict['jaccard'] = jaccard(vectorA,vectorB)
    except Exception as e:
        #print "*** jaccard exception ***  " + str(e)
        distance_dict['jaccard'] = np.nan

    try:
        distance_dict['canberra'] = canberra(vectorA, vectorB)
    except Exception as e:
        #print "*** canberra exception ***  " + str(e)
        distance_dict['canberra'] = np.nan

    try:
        distance_dict['euclidean'] = euclidean(vectorA,vectorB)
    except Exception as e:
        #print "*** euclidean exception ***  " + str(e)
        distance_dict['euclidean'] = np.nan

    try:
        distance_dict['minkowski'] = minkowski(vectorA,vectorB,3)
    except Exception as e:
        #print "*** minkowski exception ***  " + str(e)
        distance_dict['minkowski'] = np.nan

    try:
        distance_dict['braycurtis'] = braycurtis(vectorA, vectorB)
    except Exception as e:
        #print "*** braycurtis exception ***  " + str(e)
        distance_dict['braycurtis'] = np.nan

    try:
        distance_dict['skew_question1'] = skew(vectorA)
    except Exception as e:
        #print "*** skew_question1 exception ***  " + str(e)
        distance_dict['skew_question1'] = np.nan

    try:
        distance_dict['skew_question2'] = skew(vectorB)
    except Exception as e:
        #print "*** skew_question2 exception ***  " + str(e)
        distance_dict['skew_question2'] = np.nan

    try:
        distance_dict['kurtosis_question1'] = kurtosis(vectorA)
    except Exception as e:
        #print "*** kurtosis exception ***  " + str(e)
        distance_dict['kurtosis_question1'] = np.nan

    try:
        distance_dict['kurtosis_question2'] = kurtosis(vectorA)
    except Exception as e:
        #print "*** kurtosis_question2 exception ***  " + str(e)
        distance_dict['kurtosis_question2'] = np.nan

    return distance_dict


def read_data():

    data = pd.read_csv("data/test.csv")
    #var = str( data.loc[data["id"]==53,"question1"])
    #print var#cleanser(var)

    print "lock and load"

    #augmenting data with basic features
    data['len_q1'] = data.question1.apply(lambda x:len(str(x)))
    data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
    data['diff_len'] = data.len_q1 - data.len_q2
    data['len_char_q1'] = data.question1.apply(lambda x:len("".join(str(x).split())))
    data['len_char_q2'] = data.question2.apply(lambda x:len("".join(str(x).split())))
    data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
    data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
    data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).
                                                    intersection(set(str(x['question2']).lower().split()))),axis=1)
    col_basic = ['len_q1', 'len_q2', 'diff_len',
           'len_char_q1', 'len_char_q2',
           'len_word_q1', 'len_word_q2',
           'common_words']

    #Levenshtein Distance features

    data['q_ratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']),str(x['question2'])),axis=1 )
    data['w_ratio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])),
                                       axis=1)
    data['partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])),
                                       axis=1)
    data['token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])),
                                       axis=1)
    data['token_sort_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])),
                                       axis=1)

    col_levenshtein = ['q_ratio', 'w_ratio',
                   'partial_ratio', 'partial_token_set_ratio',
                   'partial_token_sort_ratio', 'token_set_ratio',
                   'token_sort_ratio']

    question_vec_cols_q1, question_vec_cols_q2 = [], []

    print "levenshtein and basic features done!"

    vector_features, col_distance = question_to_vector(data['question1'], data['question2'])
    col_distance = ['WMD_basic','WMD_normalized'] + col_distance

    print "assigning vector values"
    for i in range(0,300):
        question_vec_cols_q1.append("vec_val_"+str(i)+"_q1")
        data["vec_val_" + str(i) + "_q1"] = vector_features[:, i]
        question_vec_cols_q2.append("vec_val_" + str(i) + "_q2")
        data["vec_val_" + str(i) + "_q2"] = vector_features[:, 300+i]

    for i,key in enumerate(col_distance):
        data[key] = vector_features[:,600+i]

    data["zero_vec_check_q1"] = vector_features[:, 613]
    data["zero_vec_check_q2"] = vector_features[:, 614]

    header = ['id','is_duplicate']
    header.extend(col_basic)
    header.extend(col_levenshtein)
    header.extend(col_distance)
    header.extend(question_vec_cols_q1)
    header.extend(question_vec_cols_q2)
    header.append('zero_vec_check_q1')
    header.append('zero_vec_check_q2')
    print "writing csv"
    data.to_csv(TRANSFORMED_DATA_FILE_PATH, columns = header, sep='\t')
    print "done!"


def main():


    start_time = time.time()
    data = pd.read_csv("data/test.csv")
    print len(data)
    #read_data()


    print "--- %s Minutes ---" % ((time.time() - start_time)/60)


if __name__ == "__main__": main()

