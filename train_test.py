from model import TransformerRegressor
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from datetime import datetime
import time
import os
# from data import datatrain_load,dataval_load

from data import data1_load,data2_load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./logs/')


# prediction horizon
K = 24


#epochs
epochs = 100


# use exogenous inputs
EXOGENOUS = True

# features
if(EXOGENOUS):
    features = ['K','uvIndex','cloudCover','sunshineDuration','windBearing','humidity','temperature','hour','dewPoint']
else:
    features = ['K']


# metrics
def mad(y_pred,y_test):
    return 100 / y_test.mean() * np.absolute(y_pred - y_test).sum() / y_pred.size

def mdb(y_pred,y_test):
    return 100 / y_test.mean() * (y_pred - y_test).sum() / y_pred.size

def r2(y_pred,y_test):
    return r2_score(y_test, y_pred)

def rmsd(y_pred,y_test):
    return 100 / y_test.mean() * np.sqrt(np.sum(np.power(y_pred - y_test, 2)) / y_pred.size)

def mae(y_pred,y_test):
    return mean_absolute_error(y_test, y_pred)

def mse(y_pred,y_test):
    return mean_squared_error(y_test, y_pred)



train_loader,val_loader,lag= data1_load(32)
test_loader,label_test,y_cs= data2_load(1)
num_clo = y_cs.shape[0]
Num_f = len(features)
input_d = Num_f*lag
model = TransformerRegressor(input_dim=input_d,output_dim=1,num_heads=1,d_model=512).to(device)
criterion = nn.MSELoss().to(device)     # 忽略 占位符 索引为0.
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
val_loss = []
train_loss = []
y_preds = np.zeros(y_cs.shape)

best_loss = np.zeros(y_cs.shape[1])


def train_val(i):
    j = 0 #如果连续15次训练bestloss都没有变化则进行下一步训练
    best_test_loss = best_loss[i]
    for epoch in range(epochs):
        if j >=15:
            break
        else: j +=1
        # print("多少次没有最优模型了：",j)
        train_epoch_loss = []
        val_epoch_loss = []
        for X_train, Y_train in train_loader:
            y = Y_train[:,i]
            y = y.unsqueeze(1)
            X_tr = np.expand_dims(X_train,0)
            y_tr = np.expand_dims(y,0)
            input_xtr = X_tr.transpose(1, 0, 2) #数组转维度，(1,32,9) -->(32,1,9)
            input_ytr = y_tr.transpose(1, 0, 2) #(32,1,1)
            input_xtr = torch.tensor(input_xtr).to(device)
            input_ytr = torch.tensor(input_ytr).to(device)
            optimizer.zero_grad()
            output = model(input_xtr.to(device))#训练
            trainloss = criterion(output, input_ytr)
            trainloss.backward()
            optimizer.step()
        train_epoch_loss.append(trainloss.item())
        train_loss.append(np.mean(train_epoch_loss))
        writer.add_scalar("train_loss", np.mean(train_epoch_loss))
        for X_val, y_val in val_loader:
            y_val = y_val[:,i]
            y_val = y_val.unsqueeze(1)
            x_v = np.expand_dims(X_val, 0)
            y_v = np.expand_dims(y_val, 0)
            input_xval = x_v.transpose(1, 0, 2)
            input_yval = y_v.transpose(1, 0, 2)
            X_val = torch.tensor(input_xval).to(device)
            Y_val = torch.tensor(input_yval).to(device)
            output = model(X_val)
            valloss = criterion(output, Y_val)
            val_epoch_loss.append(valloss.item())
        val_loss.append(np.mean(val_epoch_loss))
        writer.add_scalar("val_loss", np.mean(val_epoch_loss), epoch)
        datetimeStr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(datetimeStr + "  预测分钟【" + str((15*(i+1))) + "】" + "  Epoch【" + str(epoch) +  "】 " +  "train_loss:" + str(round(np.mean(train_epoch_loss), 5)) + "  val_loss:" + str(round(np.mean(val_epoch_loss), 5)))
        # print("第 %d min 的 epoch:" % (15*(i+1)), epoch, "train_epoch_loss:", np.mean(train_epoch_loss), "val_epoch_loss:", np.mean(val_epoch_loss))
        # 保存下来最好的模型：
        if np.mean(val_epoch_loss) < best_test_loss:
            j = 0
            best_test_loss = np.mean(val_epoch_loss)
            best_model = model
            print("best_test_loss:", best_test_loss)
            torch.save(best_model.state_dict(), f'./logs/best_Transformer_trainModel_{i}.pth')

    


####test部分
def test(i):
    model.load_state_dict(torch.load(f'./logs/best_Transformer_trainModel_{i}.pth'))
    model.to(device)
    model.eval()
    y_pred = []
    for X_test,y_test in test_loader:
        y_test = y_test[:,i]
        y_test = y_test.unsqueeze(1)
        x_tt = np.expand_dims(X_test, 0)
        y_tt = np.expand_dims(y_test, 0)
        input_xtt = x_tt.transpose(1, 0, 2)
        input_ytt = y_tt.transpose(1, 0, 2)
        # print(input_ytt.shape)
        X_tt = torch.tensor(input_xtt).to(device)
        Y_tt = torch.tensor(input_ytt).to(device)
        # print(Y_tt.shape)
        output = model(X_tt)
        # output = output.cpu().detach().numpy()[0]
        y_pred.append(output)
        Y_tt = Y_tt.cpu().detach().numpy()[0] #将Y_tt从GPU移动到CPU，并释放GPU上的内存,并保存第一个值（这里本来是为了做对比留下来的，但是后面直接取值label_test，最后没用到）
    pred = [tensor.detach().cpu().numpy() for tensor in y_pred]
    pred = np.array(pred)
    pred =pred.reshape(num_clo)
    y_preds[:,i] = pred
    return y_preds

for i in range(K):
    best_loss[i] = 10000
    train_val(i)

for i in range(K):
    preds = test(i)





# transform to GHI

y_pred2 = np.multiply(preds, y_cs)

y_test2 = np.multiply(label_test,y_cs)
y_test2 =y_test2.numpy()
# save results
results = {'K':[],'Time[min]':[],'MAD[%]':[],'R2':[],'RMSD[%]':[]}
for i in range(K):
    results['K'].append(i+1)
    results['Time[min]'].append((i+1)*15)
    results['MAD[%]'].append(np.round(mad(y_pred2[:,i],y_test2[:,i]),2))
    results['R2'].append(np.round(r2(y_pred2[:,i],y_test2[:,i]),2))
    results['RMSD[%]'].append(np.round(rmsd(y_pred2[:,i],y_test2[:,i]),2))

# create results dataframe
results = pd.DataFrame(results)
results = results.set_index('K')
print(results.head(K))

