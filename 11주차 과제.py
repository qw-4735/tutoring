#%%
# 기본 데이터 정리 및 처리
import numpy as np
import pandas as pd

df_2015 = pd.read_csv('data_2015.csv')
df_2016 = pd.read_csv('data_2016.csv')
df_2017 = pd.read_csv('data_2017.csv')
df_2018 = pd.read_csv('data_2018.csv')
df_2019 = pd.read_csv('data_2019.csv')
df_2020 = pd.read_csv('data_2020.csv')
df_2021 = pd.read_csv('data_2021.csv')
df_2022 = pd.read_csv('data_2022.csv')

df=pd.concat([df_2015,df_2016,df_2017,df_2018,df_2019,df_2020,df_2021,df_2022])
df.shape #(196848, 15)

'''
ymdhm : 년월일시분
swl : 팔당댐 현재수위 (단위: El.m)
inf : 팔당댐 유입량 (단위: m^3/s)
sfw : 팔당댐 저수량 (단위: 만m^3)
ecpc : 팔당댐 공용량 (단위: 백만m^3)
tototf : 총 방류량 (단위: m^3/s)
'''

df = df.drop(['wl_1018662','fw_1018662','wl_1018683','fw_1018683','wl_1019630','fw_1019630'],axis=1)
#사용하지 않는 다른 대교 drop
df.shape #(196848, 9)
df.columns

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
#데이터 순서 바꾸기

df.describe()
df.shape #(196848, 13)

#%%
#결측값 처리
df.info() #결측값 확인 ->  swl, inf, sfw, ecpc, tototf 열 결측값 존재(196848-196126 개)
df['ymdh']=pd.to_datetime(df['ymdh'])
df=df.set_index('ymdh')
df2 = df.interpolate(method='time')
# interpolate : 결측값을 보간하는 방법
# 옵션 ‘time’: 시간/날짜 간격으로 보간. 이때 시간/날짜가 index로 되어있어야함.

df2.reset_index(drop = False, inplace = True)
df2
df2.describe() #inf, sfw,tototf의 최소값이 음수이다 


df2[df2.inf < 0] #팔당댐 유입량이 0 미만 -> 5809 rows
#%%
'''
#데이터 시각화 소요시간이 너무 길어서 추후 진행
# 데이터 시각화
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('seaborn-whitegrid')
import plotly
import plotly.graph_objects as go
import plotly.express as px
import random
line_color = ['#FFBF00','#FF7F50','#DE3163']

# 많의 양의 데이터를 줌 가능 라인 그래프 그려주는 함수
def line_slider (data, x, y):
    line_color = ['#FFBF00','#FF7F50','#DE3163']
    x = 'ymdhm'
    y = y
    fig = px.line(data, x=x, y=y, title=y)
    fig.update_traces(line_color= random.choice(line_color))
    fig.update_xaxes(
        rangeslider_visible=True)
    fig.show()

f, ax = plt.subplots(figsize=(20, 10))
sns.lineplot(x=df2['ymdh'], y=df2['swl'], ax=ax, color=random.choice(line_color))


'''


#%%
#평균 이용해서 10분 단위 데이터를 한시간 단위 데이터로 만들기
df3 = df2.groupby(['ymdh','year','month','day','hour']).mean()
df3 =df3.reset_index()

df3 # 32808 rows × 12 columns


#%%
# 데이터 분리
import torch

df3.dtypes


y = df3[['wl_1018680','ymdh']]
X = df3.drop(columns=['wl_1018680','year','month','day','hour'])
y.shape #(32808, 1)
X.shape #(32808, 7)

32808*0.8 #26246.4

X.iloc[26246-1,:] #2020-10-21 13:00:00

y_train = y.loc[y.ymdh <= '2020-10-21 13:00:00',:]
y_test = y.loc[y.ymdh >= '2020-10-21 13:00:00',:]
y_train = y_train.drop(columns = ['ymdh'])
y_test = y_test.drop(columns = ['ymdh'])

X_train = X.loc[X.ymdh <= '2020-10-21 13:00:00',:]
X_test = X.loc[X.ymdh >= '2020-10-21 13:00:00',:]

X_train.set_index('ymdh')
X_train

#%%
# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#MinMaxScaler() : 음수 값이 있으면 -1에서 1 값으로 변환

X_train_arr = X_train.copy()
X_test_arr = X_test.copy()
X_train_arr[['swl','inf','sfw','ecpc','tototf','tide_level']] = scaler.fit_transform(X_train[['swl','inf','sfw','ecpc','tototf','tide_level']])
X_train_arr

'''
X_train에 'ymdh'가 포함된 상태로 scaling하려고 하면, 'ymdh'가 datetime이기 때문에 scaling 할 수 없다는 오류가 나오는데,
'ymdh'를 제외하고 scaling을 진행해도 괜찮은 것일까요?
'''

X_test_arr[['swl','inf','sfw','ecpc','tototf','tide_level']] = scaler.transform(X_test[['swl','inf','sfw','ecpc','tototf','tide_level']])
# 학습 데이터 세트로 fit()을 수행한 결과를 이용해 transform() 변환을 적용


y_train_arr = y_train.copy()
y_test_arr = y_test.copy()
y_train_arr[['wl_1018680']] = scaler.fit_transform(y_train[['wl_1018680']])
y_test_arr[['wl_1018680']] = scaler.transform(y_test[['wl_1018680']])

'''
반드시 테스트 데이터는 학습 데이터의 스케일링 기준에 따라야 한다
따라서 테스트 데이터에 다시 fit()을 적용해서는 안 되며 
학습 데이터로 이미 fit()이 적용된 Scaler객체를 이용해 transform()으로 변환
'''

X_train.shape #(26246, 7)
X_test.shape #(6563, 7)
y_train_arr.shape #(26246, 1)
y_train_arr

#슬라이딩
#24시간으로 다음 12시간 예측하기


#이동평균선(rolling) : x이동평균선이란 (t-x)~x의 함숫값의 평균을 f(x)의 값으로 하는것을 의미

X_train_arr['12MA_swl'] = X_train_arr['swl'].rolling(24).mean().shift(12,fill_value=0)
X_train_arr['12MA_inf'] = X_train_arr['inf'].rolling(24).mean().shift(12,fill_value=0)
X_train_arr['12MA_sfw'] = X_train_arr['sfw'].rolling(24).mean().shift(12,fill_value=0)
X_train_arr['12MA_ecpc'] = X_train_arr['ecpc'].rolling(24).mean().shift(12,fill_value=0)
X_train_arr['12MA_tototf'] = X_train_arr['tototf'].rolling(24).mean().shift(12,fill_value=0)
X_train_arr['12MA_tide_level'] = X_train_arr['tide_level'].rolling(24).mean().shift(12,fill_value=0)
#rolling(24):24시간 이동평균을 구한다
#shift를 하는데 12개 단위로 shift를 한다.
#fill_value=0 : 앞에서 12개를 shift 하면 첫 12개는 값이 없을 테니, 0으로 채운다

print(y_train_arr)

'''
sliding 방식이 잘 이해가 안 가는 것 같은데,
X의 feature로 y를 예측한다고 하면, x만 sliding 시킨 값으로 예측하면 되는 것인가요?
'''

#토치 변환
X_train_arr = X_train_arr.drop(columns=['swl','inf','sfw','ecpc','tototf','tide_level'])
X_train_arr1 = X_train_arr.drop(columns='ymdh')
X_test_arr1 = X_test_arr.drop(columns='ymdh')

train_features = torch.Tensor(X_train_arr1.values)
train_targets = torch.Tensor(y_train_arr.values)
test_features = torch.Tensor(X_test_arr1.values)
test_targets = torch.Tensor(y_test_arr.values)

from torch.utils.data import TensorDataset, DataLoader
train = TensorDataset(train_features, train_targets)
test = TensorDataset(test_features, test_targets)

batch_size = 64
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

'''
26246, 6562 -> 보통 2의 배수로 설정하므로, 2의 거듭제곱.. 로 설정한다

가중치를 한 번 업데이트하기 위해 한 개 이상의 훈련 데이터 묶음을 사용하는데, 
이것을 배치라고 하고 이 묶음의 사이즈를 배치 사이즈(batch size)라고 한다.
'''
train_features.shape[0]


#%%
#RNN
import torch.nn as nn
class RNNModel(nn.Module):
    def __init__(self, sequence_length, batch_size, input_size, hidden_size,output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size

        # RNN layers
        self.rnn = nn.RNN(
            sequence_length, batch_size, input_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

'''
out, _ = self.rnn(x) 부분에서
RuntimeError: input must have 3 dimensions, got 2 가 뜨는데,
rnn input인 sequence_length, batch_size, input_size 를 다 넣었는데, 추가적으로 더 넣어야할 부분이 있을까요?
'''

input_size = 6
sequence_length = 26246
batch_size = 64
output_size = 1
#단순 예측이므로 ouput_size=1
hidden_size=output_size
model = RNNModel(sequence_length=sequence_length,
                 batch_size = batch_size,
                 input_size = input_size,
                 hidden_size = hidden_size,
                 output_size = output_size
                )

#48시간, 64개 batch sieze, 6개의 feature

'''
batch_first -> 처음 들어오는 배치의 크기에 맞추는 것
input: batch_first=False 인 경우 (L(sequence length), N(batch size), H_in(input size)) 형태의 tensor
       batch_first=True 인 경우 (N, L, H_in) 형태의 tensor
output: batch_first=False 인 경우 (L, N, D*H_out) 형태의 tensor
        batch_first=True 인 경우 (N, L, D*H_out) 형태의 tensor
h_n: (D*num_layers, N, H_out)

Sequence length : Input data의 길이
input_size: input의 feature dimension

output은 hidden_states(각 time step에 해당하는 hidden state들의 묶음)를, 
h_n은 모든 sequence를 거치고 나온 마지막 hidden state를 의미
'''

'''
hidden_state에 대해서 어떤 사이트에서는 output_size와 hidden_size 가 같아야한다고 하고(예를 들어 분류와 같은 경우 레이블 수가 hidden_size)
어떤 사이트에서는 hidden_size가 많을수록 정보의 양이 많아진다라고 표현하는데 어떤 것이 맞을까요?
many to many나 many to one의 차이라고 생각하면 될까요?
'''

#%%
import torch.optim as optim
criterion = nn.MSELoss()
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#%%


#%%
# 학습   

def train(model, criterion, loader):

    model.train()

    for step, (x_batch, y_batch) in enumerate(loader):

        optimizer.zero_grad()

        for cell, target in zip(x_batch, y_batch):
            output, hidden= model(cell)
        loss = criterion(output.squeeze(0), target)
        loss.backward()
        optimizer.step()
        
    return loss



# 평가
@torch.no_grad()
def evaluate(model, criterion, loader):

    model.eval() 
    
    test_loss = 0 
    for step, (x_batch, y_batch) in enumerate(loader):

        for cell, target in zip(x_batch, y_batch):
            cell = cell.unsqueeze(0)
            output, hidden = model(cell) 
        loss = criterion(output.squeeze(0), target)
        test_loss += loss
              
    test_loss /= len(list(loader))

    return test_loss

#%%
# 실행
import tqdm
import random
random.seed(1)

for epoch in tqdm.tqdm(range(20)):

    train(model, criterion, train_loader)
    train_loss = evaluate(model, criterion, train_loader)
    val_loss = evaluate(model, criterion, test_loader)
    
    print(f'epoch: {epoch}, train loss: {train_loss}, val_loss : {val_loss}')


# test
test_loss = evaluate(model, criterion, test_loader)
print(f"test loss : {test_loss}")