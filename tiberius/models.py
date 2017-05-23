#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: quora learning models

"""
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score
import xgboost
import cPickle
import sys

try:
    file = sys.argv[1]
    file_name = sys.argv[2]
    PICLKLE_FILE_PATH = "data/train/glove-xgboost.pkl"
    SUBMISSION_FILE_PATH = "data/"+str(file)+"/"+str(file_name)+".csv"
    TRANSFORMED_DATA_FILE_PATH = "data/"+str(file)+"/"+str(file)+"_transformed.csv"
except Exception as e:
    print str(e)
    print "arguments error"
    exit(0)

def prototype():

    data = pd.read_csv(TRANSFORMED_DATA_FILE_PATH)
    y = np.array(data.iloc[:,2])
    # 628 for glove, 630 for word2vec including the WMDs
    X = np.array(data.iloc[:,3:628])
    print "loaded"

    X[np.isnan(X)] = 0#-5555555
    X[np.isinf(X)] = 0#-5555555

    print "loading done..."
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    #XGboost_model(X, None, y, None)
    XGboost_model_pickle(X, None, y, None)

def sequential_reader():

    start = 0
    stop = 2345796
    step = 10000

    with open(PICLKLE_FILE_PATH, 'rb') as pkl:
        clf = cPickle.load(pkl)

    with open(SUBMISSION_FILE_PATH, 'a') as submission_file:
        submission_file.write('test_id,is_duplicate' + '\n')

        for i in range(start,stop,step):
            print "i :",i
            data = pd.read_csv(TRANSFORMED_DATA_FILE_PATH, skiprows=i,nrows=step)
            # 628 for glove, 630 for word2vec including the WMDs
            X = np.array(data.iloc[:, 3:628])
            X[np.isnan(X)] = 0#-5555555
            X[np.isinf(X)] = 0#-5555555
            result = clf.predict_proba(X)

            for j in range(0,step):
                submission_file.write(str(i + j) + ',' + str('%.1f'%result[j][1]) + '\n')

        print "last i: ",i
        last_batch = stop - i
        data = pd.read_csv(TRANSFORMED_DATA_FILE_PATH, skiprows=i, nrows=last_batch)
        #628 for glove, 630 for word2vec including the WMDs
        X = np.array(data.iloc[:, 3:628])
        X[np.isnan(X)] = 0#-5555555
        X[np.isinf(X)] = 0#-5555555
        result = clf.predict_proba(X)
        for j in range(0, last_batch):
            submission_file.write(str(i + j) + ',' + str('%.1f'%result[j][1]) + '\n')


def XGboost_model_pickle(X_train, X_test, y_train, y_test):
    clf = xgboost.XGBClassifier(learning_rate=0.1,
                                n_estimators=251,
                                max_depth=9,
                                min_child_weight=1,
                                gamma=0.2,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                objective='binary:logistic',
                                nthread=4,
                                scale_pos_weight=1,
                                seed=27)

    clf.fit(X_train,y_train)
    with open(PICLKLE_FILE_PATH, 'wb') as fid:
        cPickle.dump(clf, fid)


def XGboost_model(X_train, X_test, y_train, y_test):

    # fit model on training data
    clf = xgboost.XGBClassifier(learning_rate=0.1,
                                n_estimators=351,
                                max_depth=9,
                                min_child_weight=1,
                                gamma=0.2,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                objective='binary:logistic',
                                nthread=4,
                                scale_pos_weight=1,
                                seed=27)

    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print scores
    print("Accuracy: %.2f%%" % (np.mean(scores) * 100.0))

    #clf.fit(X_train,y_train)
    #print "fitting done"
    #results = clf.predict(X_test)
    #score = accuracy_score(y_test,results)
    #print "Accuracy: ",score

def main():

    start_time = time.time()
    sType = sys.argv[1]

    if sType == 'train':
        prototype()
    if sType == 'test':
        sequential_reader()

    print "--- %s Minutes ---" % ((time.time() - start_time)/60)


if __name__ == "__main__": main()