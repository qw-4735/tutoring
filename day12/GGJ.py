import numpy as np
import pandas as pd
import tqdm
import random
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader 
import torch.optim as optim
#%%

df_2015 = pd.read_csv('D:\Internship\day9\water_data\data_2015.csv')
df_2016 = pd.read_csv('D:\Internship\day9\water_data\data_2016.csv')
df_2017 = pd.read_csv('D:\Internship\day9\water_data\data_2017.csv')
df_2018 = pd.read_csv('D:\Internship\day9\water_data\data_2018.csv')
df_2019 = pd.read_csv('D:\Internship\day9\water_data\data_2019.csv')
df_2020 = pd.read_csv('D:\Internship\day9\water_data\data_2020.csv')
df_2021 = pd.read_csv('D:\Internship\day9\water_data\data_2021.csv')
df_2022 = pd.read_csv('D:\Internship\day9\water_data\data_2022.csv')

df=pd.concat([df_2015,df_2016,df_2017,df_2018,df_2019,df_2020,df_2021,df_2022])

df = df.drop(['wl_1018662','fw_1018662','wl_1018683','fw_1018683','wl_1019630','fw_1019630'],axis=1)

#연, 월, 일, 시, 분으로 나누기
df['ymdhm'] = df['ymdhm'].astype('str')
df['ymdhm'].dtype

date_list = df['ymdhm'].str.split('-')
df['year'] = date_list.str.get(0)
df['month'] = date_list.str.get(1)
df['dhm'] =date_list.str.get(2)

date_list2 = df['dhm'].str.split(' ')
df['day'] = date_list2.str.get(0)
df['hm'] = date_list2.str.get(1)

date_list3 = df['hm'].str.split(':')
df['hour'] = date_list3.str.get(0)
df['minute'] = date_list3.str.get(1)


df = df.drop(['ymdhm','dhm','hm','fw_1018680'],axis=1)
df['ymdh'] = df['year'] + '-'+ df['month'] + '-'+ df['day'] + ' '+ df['hour']
df = df[['ymdh','year', 'month', 'day', 'hour', 'minute','swl', 'inf', 'sfw', 'ecpc', 'tototf', 'tide_level', 'wl_1018680']]


#결측값 처리
df.info() #결측값 확인 ->  swl, inf, sfw, ecpc, tototf 열 결측값 존재(196848-196126 개)
df['ymdh']=pd.to_datetime(df['ymdh'])
df=df.set_index('ymdh')

df2 = df.interpolate(method='time')

#평균 이용해서 10분 단위 데이터를 한시간 단위 데이터로 만들기
df3 = df2.groupby(['ymdh','year','month','day','hour']).mean()
df3 =df3.reset_index()

df4 = df3.drop(columns=['year', 'month', 'day', 'hour'])
df4 = df4.set_index('ymdh')

#df4.shape  # (32808, 7)

#%%
seq_length = 24   


# 1) train_test split
train_size = int(len(df4)*0.7)

train_set = df4[0:train_size]   # (22965, 7)
test_set = df4[train_size-seq_length: ]   # (9867, 7 )   # 첫번째 seq_length를 train_set에서 가져옴



# 2) scaling
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)


# 3) sliding window (sequence data 만들기)
def build_dataset(data, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(data)-seq_length):
        _x = data[i:i+seq_length, :]  # 0 : 0+24   # 0: 0+48  # (7,5)
        _y = data[i+seq_length ,[-1] ]  # # 48:48+12    # data[i+seq_length , [-1]]
        # print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)

    return np.array(dataX), np.array(dataY)



trainX, trainY = build_dataset(np.array(train_scaled), seq_length)  # (22941, 24, 7) , (22941, 1)  # (num_data, seq_length, input_dim), (num_data, ?)
testX, testY = build_dataset(np.array(test_scaled), seq_length)   # (9843, 24, 7) , (9843, 1)
#trainX.shape
#trainY.shape
#testX.shape
#testY.shape


# 4) 텐서로 변환
trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)

testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

train_dataset = TensorDataset(trainX_tensor, trainY_tensor)
test_dataset = TensorDataset(testX_tensor, testY_tensor)



# 5) DataLoader에 입력
train_loader = DataLoader(train_dataset, batch_size = 64 , shuffle =False, drop_last= True)
test_loader = DataLoader(test_dataset, batch_size = 64 , shuffle =False, drop_last= True)

#next(iter(train_loader))[0].shape  # torch.Size([64, 24, 7])
#next(iter(train_loader))[1].shape  # torch.Size([64, 1])



#%%
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def initHidden(self):
        return  torch.zeros(self.num_layers, 64, self.hidden_size)  # 1 -> batch_size(64) 
                #torch.zeros(self.num_layers, 1, self.hidden_size)
    
    def forward(self, input):
        hidden = self.initHidden()
        output, _ = self.rnn(input, hidden)
        output = output[:,-1,:]
        output = self.fc(output)
        return output    
    
#%%
input_size = 7
hidden_size = 10
output_size = 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = RNN(input_size, hidden_size, output_size, 2).to(device)    


learning_rate = 0.005
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr= learning_rate)



   
def train(model,  criterion, train_loader):
    
    model.train()
    
    for step, (x_batch, y_batch) in enumerate(train_loader):
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        
        optimizer.step()
        
    return loss

train(model, criterion, train_loader)        
        
@torch.no_grad()        
def evaluate(model, criterion, loader):
    
    model.eval()
    
    test_loss = 0
    for step, (x_batch, y_batch) in enumerate(loader):
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        y_pred = model(x_batch)
        
        loss = criterion(y_pred, y_batch)
        test_loss += loss
    
    test_loss /= len(list(loader))
    
    return test_loss    
          


random.seed(1)

for epoch in tqdm.tqdm(range(10)):
    
    train(model, criterion, train_loader) 
    train_loss = evaluate(model, criterion, train_loader)
    test_loss = evaluate(model, criterion, test_loader)
    
    print(f'epoch : {epoch}, train_loss:{train_loss}, test_loss:{test_loss}')
        
        
