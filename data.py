# %matplotlib inline
from datetime import datetime
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import os
# disable CPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# lagged GHI values
LAG = 9

# prediction horizon
K = 24

EXOGENOUS = True

# features
if(EXOGENOUS):
    features = ['K','uvIndex','cloudCover','sunshineDuration','windBearing','humidity','temperature','hour','dewPoint']
else:
    features = ['K']

df = pd.read_csv("./datasets/clean_dataset.csv",header=0, index_col=0, parse_dates=True).sort_index()
df_GHI = df[['K']].copy()
# create exogenous regressors
for feature in features:
    df_GHI[feature] = df[feature]
    for i in range(LAG-1):
        df_GHI[feature+'-'+str(i+1)] = df[feature].shift(i+1)
# create target values
for i in range(1,K+1):
    #  df_GHI['K+'+str(i)] = df['K'].shift(-i)
    df_k= df['K'].shift(-i).rename('K+'+str(i))
    df_GHI = pd.concat([df_GHI,df_k],axis=1)
    
# create clear sky target values
for i in range(1,K+1):
    # df_GHI['GHI_cs+'+str(i)] = df['GHI_cs'].shift(-i)
    df_cs = df['GHI_cs'].shift(-i).rename('GHI_cs+'+str(i))
    df_GHI = pd.concat([df_GHI,df_cs],axis=1)

# drop nan due to shifting
df_GHI = df_GHI.dropna()
    # create training set
X_train = df_GHI['2010-1-1':'2014-6-30'].values[:,:-K*2]
y_train = df_GHI['2010-1-1':'2014-6-30'].values[:,-K*2:-K]

# create validation set
X_val = df_GHI['2014-7-1':'2014-12-31'].values[:,:-K*2]
y_val = df_GHI['2014-7-1':'2014-12-31'].values[:,-K*2:-K]
# create test set
X_test = df_GHI['2015-1-1':'2015-12-31'].values[:,:-K*2]
y_test = df_GHI['2015-1-1':'2015-12-31'].values[:,-K*2:-K]
# get clear sky target values
y_cs = df_GHI['2015-1-1':'2015-12-31'].values[:,-K:]
# scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


def data1_load(batch):
    data_train = torch.from_numpy(X_train).type(torch.float32)
    label_train = torch.from_numpy(y_train).type(torch.float32)
    data_val = torch.from_numpy(X_val).type(torch.float32)
    label_val = torch.from_numpy(y_val).type(torch.float32)
    # data_test = torch.from_numpy(X_test).type(torch.float32)
    # label_test = torch.from_numpy(y_test).type(torch.float32)

    dataset_train = TensorDataset(data_train,label_train)
    datatrain_loader = DataLoader(dataset_train,batch_size=batch,shuffle=False)  # 数据迭代器DataLoader  
    dataset_val = TensorDataset(data_val,label_val)
    dataval_loader = DataLoader(dataset_val,batch_size=batch,shuffle=False)
    # dataset_test = TensorDataset(data_test,label_test)
    

    return datatrain_loader,dataval_loader,LAG


def data2_load(batchtest):
    data_test = torch.from_numpy(X_test).type(torch.float32)
    label_test = torch.from_numpy(y_test).type(torch.float32)

   
    dataset_test = TensorDataset(data_test,label_test)
    datatest_loader = DataLoader(dataset_test,batch_size=batchtest,shuffle=False)
    return datatest_loader,label_test,y_cs