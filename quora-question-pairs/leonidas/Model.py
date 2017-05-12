import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import xgboost as xg
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.manifold import t_sne
from sklearn.model_selection import train_test_split
import time

def Xgb_Boost_Model():

    dataframe=pd.read_csv('Variant1.csv')
    dataframe.replace(np.nan,-5555555)
    dataframe.replace(np.inf,-5555555)
    data = np.array(dataframe)
    target=data[:,-1]
    Features=data[:,2:628]


    xg_boost=xg.XGBClassifier(learning_rate =0.1,
             n_estimators=251,
             max_depth=10,
             min_child_weight=1,
             gamma=0.5,
             subsample=0.8,
             colsample_bytree=0.8,
             objective= 'binary:logistic',
             nthread=-1,
             scale_pos_weight=1,
             seed=27)

    x_train,x_test,y_train,y_test=train_test_split(Features,target,test_size=0.3,stratify=True)
    xg_boost.fit(x_train,y_train)
    y_pred=xg_boost.predict(x_test)
    print accuracy_score(y_test,y_pred)

   

def Main():


    print("Programe started....")
    start_time = time.time()
    Xgb_Boost_Model()
    print("--- %s seconds ---" % (time.time() - start_time))

Main()
