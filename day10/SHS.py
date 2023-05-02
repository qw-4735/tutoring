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
from sklearn.model_selection import train_test_split

df3.dtypes

y = df3[['wl_1018680']]
X = df3.drop(columns=['ymdh', 'wl_1018680'])
X = X.astype('float')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)


# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#MinMaxScaler() : 음수 값이 있으면 -1에서 1 값으로 변환

X_train_arr = scaler.fit_transform(X_train)
X_test_arr = scaler.transform(X_test)
# 학습 데이터 세트로 fit()을 수행한 결과를 이용해 transform() 변환을 적용

y_train_arr = scaler.fit_transform(y_train)
'''
array([[0.02181208],
       [0.02274422],
...
       [0.02181208],
       ...,
       [0.12956749],
       [0.12453393],
       [0.10123043]])
'''

y_test_arr = scaler.transform(y_test)
'''
반드시 테스트 데이터는 학습 데이터의 스케일링 기준에 따라야 한다
따라서 테스트 데이터에 다시 fit()을 적용해서는 안 되며 
학습 데이터로 이미 fit()이 적용된 Scaler객체를 이용해 transform()으로 변환
'''

X_train.shape #(26246, 10)
X_test.shape #(6562, 10)
y_train_arr.shape #(26246, 1)

train_features = torch.Tensor(X_train_arr)
train_targets = torch.Tensor(y_train_arr)
test_features = torch.Tensor(X_test_arr)
test_targets = torch.Tensor(y_train_arr)

from torch.utils.data import TensorDataset, DataLoader
train = TensorDataset(train_features, train_targets)
test = TensorDataset(test_features, test_targets)

batch_size = 2
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

'''
26246, 6562 -> 보통 2의 배수로 설정하므로, 2.. 로 설정한다(?)


가중치를 한 번 업데이트하기 위해 한 개 이상의 훈련 데이터 묶음을 사용하는데, 
이것을 배치라고 하고 이 묶음의 사이즈를 배치 사이즈(batch size)라고 한다.
'''
