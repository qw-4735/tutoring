#%%
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset 
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random

import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# %%
df_2015 = pd.read_csv('data_2015.csv')
df_2016 = pd.read_csv('data_2016.csv')
df_2017 = pd.read_csv('data_2017.csv')
df_2018 = pd.read_csv('data_2018.csv')  
df_2019 = pd.read_csv('data_2019.csv')
df_2020 = pd.read_csv('data_2020.csv')
df_2021 = pd.read_csv('data_2021.csv')
df_2022 = pd.read_csv('data_2022.csv')

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
df4.head()

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
        _x = data[i:i+seq_length, :]  # 0 : 0+24   # 1: 1+24  # (7,5)
        _y = data[i+seq_length ,[-1] ]  # 0+24    # data[i+seq_length , [-1]]
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

#%%
iw=24
ow=1
train_loader = DataLoader(train_dataset, batch_size = 64 , shuffle =False, drop_last= True)
test_loader = DataLoader(test_dataset, batch_size = 64 , shuffle =False, drop_last= True)


#next(iter(train_loader))[0].shape  # torch.Size([64, 24, 7])
#next(iter(train_loader))[1].shape  # torch.Size([64, 1])

# %%
#tranformer에서 encoder만 사용!
class TFModel(nn.Module):
    def __init__(self, num_feat, iw, ow, d_model, nhead, nlayers, dropout=0.5):
        '''
        num_feat = x_train.shape[1] #21
        iw = 24
        ow = 1
        d_model = 64
        nhead = 2
        nlayers = 2
        dropout = 0.1
        '''
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        '''
        nn.TransformerEncoderLayer : self-attn(MultiheadAttention)과 feedforward로 구성
        *d_model : Encoder에서 input/output의 차원. Embedding vector의 크기도 d_model 
        *nhead: multiheadattention에서 head의 갯수로, 벡터를 nhead만큼 나누어 병렬로 attention을 진행
        dim_feedforward: transformer 내부 FF의 hidden 차원 (default=2048)
        '''
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(num_feat, d_model//2),
            nn.LeakyReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LeakyReLU(),
            nn.Linear(d_model//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.LeakyReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        #torch.triu() : 상삼각행렬로 만들기
        '''트랜스포머는 한번에 보기 때문에 뒤의 값을 가려야 한다.
        그래서 과거부터 현재 보는 값까지는 1, 뒤에는 0을 넣는다.'''
        #transpose(0, 1) #행과 열의 위치를 바꾸기
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        # output = self.transformer_encoder(src, srcmask)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).to(torch.float32) * (-math.log(10000.0) / d_model))
        
        # (i가 짝수일때 : sin, 홀수일때 : cos)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        #pe[:, :x.size(0)] : x의 shape에 맞출 수 있도록 indexing
        return self.dropout(x)

# %%
lr = 1e-3
num_feat = trainX.shape[2]
d_model = 64
nhead = 2
iw =24
ow=1
nlayers = 2
dropout = 0.1

model = TFModel(num_feat, iw, ow, d_model, nhead, nlayers, dropout).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %%
epoch = 100
model.train()

'''
for b in train_loader:
    inputs, outputs = b
    break
outputs = outputs.unsqueeze(2)
inputs.size() #torch.Size([64, 24, 7])
outputs.size() #torch.Size([64, 1, 1])
'''

progress = tqdm(range(epoch))
for i in progress:
    batchloss = 0.0
    for (inputs, outputs) in train_loader:
        outputs = outputs.unsequeeze(2)
        optimizer.zero_grad()
        src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
        #Self-attention을 훈련시킬 때 Attend 하지 말아야할 토큰을 가리는 역할
        #출력 토근은 현재보다 오른쪽 (미래) 토큰은 Attend 하지 못하도록 한다
        result = model(inputs, src_mask) # torch.Size([64, 120])
        loss = criterion(result, outputs[:,:,0].float().to(device))
        loss.backward()
        optimizer.step()
        batchloss += loss
    progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))
# torch.save(model.state_dict(), 'model.pth')

#%%
@torch.no_grad()        
def evaluate():
    model.eval()
    
    inputs = torch.tensor(train_scaled[-iw:]).reshape(1,-1,num_feat).to(torch.float32).to(device)
    src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
    pred = model(inputs, src_mask)
    
    return pred.detach().cpu().numpy()

#%%
