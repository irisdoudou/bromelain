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

train_df=pd.read_csv("C:/Users/Administrator/Desktop/kaggle/flight-delays-fall-2018/flight_delays_train.csv")
train_df.head()
test_df=pd.read_csv("C:/Users/Administrator/Desktop/kaggle/flight-delays-fall-2018/flight_delays_test.csv")
test_df.head()

train_df_X=train_df.loc[:,'Month':'Distance']
train_df_X.head()
train_df_Y=train_df.loc[:,'dep_delayed_15min']
train_df_Y
#先把train的自变量与因变量分开
#然后对test 和train 进行concat这样可以一起处理
#concat
data=pd.concat([train_df_X,test_df])
#改数据格式
#month
data.replace(np.nan, 0, inplace=True)  #这里缺失值直接用0填补 也可以用热卡啥的
data.replace(np.inf, 0, inplace=True)
data['Month']=data['Month'].str.split('-').str.get(1)#####去掉前面的符号

data['Month']=data['Month'].astype('int64')
#dayofmonth
data['DayofMonth']=data['DayofMonth'].str.split('-').str.get(1)
data['DayofMonth']=data['DayofMonth'].astype('int64')
#dayofweek
data['DayOfWeek']=data['DayOfWeek'].str.split('-').str.get(1)
data['DayOfWeek']=data['DayOfWeek'].astype('int64')
data.info()
#onehotencoder避免了数值问题 比如1类和3类的平均点在2
print(data['UniqueCarrier'].value_counts().shape) #23个航空公司
print(data['Origin'].value_counts().shape)
print(data['Dest'].value_counts().shape) #307个机场
#onehotencoder

#但是出发和到达的特征是一样的 这样会不会导致往返航班的特征一样呢
#不会 dest和dep两个连名字都不一样
import pandas as pd
data_dummy=pd.get_dummies(data)
origin=pd.factorize(pd.Series(data['Origin']))
data_dummy.info()  #瞬间变大  
data_dummy.shape
data_dummy.columns
#日期时间全部转化为数值型 还有航空公司和出发和到达三个特征 可以用onehot和labelencoder
type(data_dummy)
#将训练集与测试集拆分
#分组后得到的是series 记得用dataframe函数进行转化
train_dummy=data_dummy[0:80000][:]
valid_dummy=data_dummy[80000:100000][:]

train_dummy.shape
y_train_1=pd.get_dummies(train_df['dep_delayed_15min'])
y_train_1=y_train_1['Y']
y_train=y_train_1[:80000][:]
y_valid=y_train_1[80000:100000][:]
y_train.shape
test_dummy=data_dummy[100000:][:]
test_dummy.shape


#建模
from sklearn.svm import LinearSVC
svc=LinearSVC(max_iter=2000).fit(train_dummy,y_train)  #太大了 跑好久哦
#max_iter设置过大会很慢 默认的又没有收敛 
svc.score(train_dummy,y_train)
svc.score(valid_dummy,y_valid)
pd=svc.predict(test_dummy)
pd.shape

#ESSENTIAL DATA ANALYSIS
train_df['dep_delayed_15min'].value_counts(normalize=True)
#normalize=True计算计数占比


