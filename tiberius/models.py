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

def read_file():

    data = pd.read_csv('data/wiki/engineered_features.csv', delimiter='\t')
    y = data.iloc[:,2]

    X = np.zeros(shape=(len(y),628))
    X = data.iloc[:,3:632]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42, stratify=y)
    stratified_sample = np.zeros(shape=(len(X_test), X_test.shape[1]+1))
    stratified_sample[:, 0:X_test.shape[1]] = X_test
    stratified_sample[:, X_test.shape[1]] = y_test
    np.save("prototype.csv", stratified_sample)

    #X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.1, random_state=42, stratify=y_test)

def prototype():

    data = pd.read_csv('data/wiki/new_train.csv', delimiter='\t')
    y = np.array(data.iloc[:,2])
    X = np.array(data.iloc[:,3:630])
    print "loaded"

    #data = np.load('data/prototype.csv.npy')
    #y = data[:,629]
    #X = data[:,0:629]

    X[np.isnan(X)] = -5555555
    X[np.isinf(X)] = -5555555

    print "loading done..."
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    #XGboost_model(X, None, y, None)
    XGboost_model_pickle(X, None, y, None)

def sequential_reader():

    start = 0
    stop = 2345796
    step = 10000

    with open('data/wiki/models/xgboost_trained.pkl', 'rb') as pkl:
        clf = cPickle.load(pkl)

    with open('data/wiki/submissions/submission-3.csv', 'a') as submission_file:
        submission_file.write('test_id,is_duplicate' + '\n')

        for i in range(start,stop,step):
            print "i :",i
            data = pd.read_csv('data/wiki/new_test.csv', skiprows=i,nrows=step, delimiter='\t')
            X = np.array(data.iloc[:, 3:630])
            X[np.isnan(X)] = -5555555
            X[np.isinf(X)] = -5555555
            result = clf.predict_proba(X)

            for j in range(0,step):
                submission_file.write(str(i + j) + ',' + str('%.1f'%result[j][1]) + '\n')

        print "last i: ",i
        last_batch = stop - i
        data = pd.read_csv('data/wiki/new_test.csv', skiprows=i, nrows=last_batch, delimiter='\t')
        X = np.array(data.iloc[:, 3:630])
        X[np.isnan(X)] = -5555555
        X[np.isinf(X)] = -5555555
        result = clf.predict_proba(X)
        for j in range(0, last_batch):
            submission_file.write(str(i + j) + ',' + str('%.1f'%result[j][1]) + '\n')



def LR_model(X_train, X_test, y_train, y_test):

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print "fitting done..."
    results = clf.predict(X_test)
    score = accuracy_score(y_test, results)
    print "predicting done ..."
    # scores = cross_val_score(LR, X, y, cv=10, scoring='roc_auc')

    print "Accuracy: ", score

def XGboost_model_pickle(X_train, X_test, y_train, y_test):
    clf = xgboost.XGBClassifier(learning_rate=0.1,
                                n_estimators=249,
                                max_depth=9,
                                min_child_weight=1,
                                gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                objective='binary:logistic',
                                nthread=-1,
                                scale_pos_weight=1,
                                seed=27)

    clf.fit(X_train,y_train)
    with open('data/wiki/models/xgboost_trained-2.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)

def XGboost_model(X_train, X_test, y_train, y_test):

    # fit model on training data
    clf = xgboost.XGBClassifier(learning_rate=0.1,
                                n_estimators=249,
                                max_depth=9,
                                min_child_weight=1,
                                gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                objective='binary:logistic',
                                nthread=-1,
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
    #read_file()
    #prototype()
    sequential_reader()
    print "--- %s Minutes ---" % ((time.time() - start_time)/60)


if __name__ == "__main__": main()