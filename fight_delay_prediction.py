# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:10:53 2019

@author: IrisDOU
"""

import warnings
import numpy as np
import pandas as pd
import pathlib as path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import seaborn as sns

data=pd.read_csv("...data.csv")
data.dropna(subset=['DEP_TIME','DEP_DELAY',
                  'ARR_TIME','ARR_DELAY','CRS_ELAPSED_TIME','TAIL_NUM'],inplace=True) 
data.info()
data.head()
data.isnull().sum()

partly=data.loc[(data["DEST_AIRPORT_ID"] == 12478)&(data["MONTH"]==1)&(data["DAY_OF_MONTH"]==1), ["ARR_DELAY"]]
partly.index = range(1, len(partly) + 1)
partly.columns
sns.lineplot(data=partly,color='b')
plt.ylabel('arrival delay/min')
plt.xlabel('flights')
plt.xticks([])
#clean variables
data['CRS_DEP_TIME']=data['CRS_DEP_TIME'].str.split(':').str.get(0)
data['CRS_DEP_TIME']=pd.DataFrame(data['CRS_DEP_TIME'],dtype=np.float)
data['DEP_TIME']=data['DEP_TIME'].str.split(':').str.get(0)
data['DEP_TIME']=pd.DataFrame(data['DEP_TIME'],dtype=np.float)
#
data['CRS_ARR_TIME']=data['CRS_ARR_TIME'].str.split(':').str.get(0)
data['CRS_ARR_TIME']=pd.DataFrame(data['CRS_ARR_TIME'],dtype=np.float)
data['ARR_TIME']=data['ARR_TIME'].str.split(':').str.get(0)
data['ARR_TIME']=pd.DataFrame(data['ARR_TIME'],dtype=np.float) 
#train_set
from sklearn.utils import shuffle
data=shuffle(data)
data.info()
data['DEP_DELAY'].head(20)
data_target=pd.DataFrame(data['ARR_DEL15'])

data_target.apply(pd.value_counts)
data_target.astype(np.bool)
data[['YEAR','QUARTER','MONTH','DAY_OF_MONTH','DAY_OF_WEEK',
     'DISTANCE']]=data[['YEAR','QUARTER','MONTH','DAY_OF_MONTH',
     'DAY_OF_WEEK','DISTANCE']].astype(float)  #将int变量转化为float

data=data.drop(['OP_CARRIER_FL_NUM','DEP_TIME','DEP_DELAY','DEP_DELAY_NEW',
                'DEP_DEL15','ARR_TIME','ARR_DELAY','ARR_DELAY_NEW','CANCELLED',
                'CANCELLATION_CODE','CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME',
                'CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY',
                'LATE_AIRCRAFT_DELAY'],axis=1)  #去除无关变量
data=data.drop(['FLIGHTS','YEAR'],axis=1)
data=data.drop(['ARR_DEL15'],axis=1)
data=data.drop(['FL_DATE'],axis=1)
categorical_features_indices = np.where(data.dtypes != np.float)[0]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data,data_target,test_size=0.3, 
                                               stratify=data_target,random_state=83)
y_test.apply(pd.value_counts)
y_train.apply(pd.value_counts)
#cat_boost model establishment
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=500,  
                           depth = 10,
                           learning_rate =0.15,
                           one_hot_max_size=31,
                           loss_function='Logloss',
                           logging_level='Verbose',
                           custom_loss='AUC',
                           eval_metric='AUC',
                           rsm = 0.78,
                           od_wait=150,
                           metric_period = 400,
                           l2_leaf_reg = 9,
                           random_seed = 967)

model.fit(X_train,y_train,plot=True,cat_features=categorical_features_indices)
import matplotlib.pyplot as plt 
fea_ = model.feature_importances_  #feature importance plot
fea_name = model.feature_names_
plt.figure(figsize=(10, 10))
plt.barh(fea_name,fea_,height =0.5)

#AUC-ROC curve/FPR-TPR curve
from catboost.utils import get_roc_curve
import sklearn
from sklearn import metrics
from catboost import Pool
eval_pool = Pool(X_test, y_test, cat_features=categorical_features_indices)
eval_train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
curve = get_roc_curve(model, eval_train_pool)
(fpr, tpr, thresholds) = curve# fpr,tpr及其相对应的阈值  是一系列对应的
roc_auc = sklearn.metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 8))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc, alpha=0.5)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.legend(loc="lower right", fontsize=16)
plt.show()

from catboost.utils import get_fpr_curve
from catboost.utils import get_fnr_curve
(thresholds, fpr) = get_fpr_curve(curve=curve)
(thresholds, fnr) = get_fnr_curve(curve=curve)
plt.figure(figsize=(16, 8))
lw = 2
plt.plot(thresholds, fpr, color='blue', lw=lw, label='FPR', alpha=0.5)
plt.plot(thresholds, fnr, color='green', lw=lw, label='FNR', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.xlabel('Threshold', fontsize=16)
plt.ylabel('Error Rate', fontsize=16)
#plt.title('FPR-FNR curves', fontsize=20)
plt.legend(loc="lower left", fontsize=16)
plt.show()
#find threshold
from catboost.utils import select_threshold
print(select_threshold(model=model, data=eval_train_pool, FNR=0.2))
print(select_threshold(model=model, data=eval_train_pool, FPR=0.4))
#confusion matrix
print(get_confusion_matrix(model, data=eval_pool))
from catboost.utils import get_confusion_matrix
#result show
test_pool=Pool(X_test,y_test,cat_features=categorical_features_indices)
from catboost import Pool
model.get_all_params()  #params

model.eval_metrics(data=eval_pool,metrics='Recall')
model.score(test_pool)
result=model.predict_proba(eval_test_pool)   
