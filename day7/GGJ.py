# 1. https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b 
# data : https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
# fitting the RNN model constructed in day6 (모델 구현할 때 line by line으로 차원 계산해보고 주석 달 것)
#%%
import plotly.graph_objs as go
from plotly.offline import iplot
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# 데이터 준비

def plot_dataset(df, title):
    data = []
    value = go.Scatter(
        x=df.index,
        y=df.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)



df = pd.read_csv('AEP_hourly.csv')
df = df.set_index(['Datetime'])
df.index = pd.to_datetime(df.index)
if not df.index.is_monotonic:
    df = df.sort_index()

df = df.rename(columns={'AEP_MW': 'value'})


def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n
    
input_dim = 100

df_generated = generate_time_lags(df, input_dim)
df_generated


# Generating features from timestamp
df_features = (
                df
                .assign(hour = df.index.hour)
                .assign(day = df.index.day)
                .assign(month = df.index.month)
                .assign(day_of_week = df.index.dayofweek)
                .assign(week_of_year = df.index.week)
              )

df_features

# One-Hot Encoding
def onehot_encode_pd(df, col_name):
    dummies = []
    for column in col_name:
        dummy = pd.get_dummies(df[column], prefix=column)
        dummies.append(dummy)
    cat_dummies = pd.concat(dummies, axis=1)
    return pd.concat([df, cat_dummies], axis=1)

df_features = onehot_encode_pd(df_features, ['month','day','day_of_week','week_of_year'])
df_features


# 주기적 시간 특성 생성
def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
             }
    return df.assign(**kwargs).drop(columns=[col_name])

df_features = generate_cyclical_features(df_features, 'hour', 24, 0)
df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 0)
df_features = generate_cyclical_features(df_features, 'month', 12, 1)
df_features = generate_cyclical_features(df_features, 'week_of_year', 52, 0)

df_features

# 휴일 추가
from datetime import date
import holidays

us_holidays = holidays.US()

def is_holiday(date):
    date = date.replace(hour = 0)
    return 1 if (date in us_holidays) else 0

def add_holiday_col(df, holidays):
    return df.assign(is_holiday = df.index.to_series().apply(is_holiday))


df_features = add_holiday_col(df_features, us_holidays)
df_features


# 데이터 분리
from sklearn.model_selection import train_test_split

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, 'value', 0.2)


# Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_arr = scaler.fit_transform(X_train)
X_val_arr = scaler.transform(X_val)
X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
y_test_arr = scaler.transform(y_test)



# dataloader
from torch.utils.data import TensorDataset, DataLoader

batch_size = 64

train_features = torch.Tensor(X_train_arr)
train_targets = torch.Tensor(y_train_arr)
val_features = torch.Tensor(X_val_arr)
val_targets = torch.Tensor(y_val_arr)
test_features = torch.Tensor(X_test_arr)
test_targets = torch.Tensor(y_test_arr)


train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)



# RNN 구축

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
       
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined) 
        output = self.i2o(combined) 
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size) 


n_input = X_train.shape[1]  # 113
n_hidden = 128
n_output = y_train.shape[1]  #1



import torch.optim as optim

model = RNN(n_input, n_hidden, n_output)
criterion = nn.MSELoss()
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 학습   

def train(model, criterion, loader):

    model.train()

    for step, (x_batch, y_batch) in enumerate(loader):

        hidden = model.initHidden()
        optimizer.zero_grad()

        for cell, target in zip(x_batch, y_batch):
            cell = cell.unsqueeze(0)
            output, hidden = model(cell, hidden)
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
        
        hidden = model.initHidden()

        for cell, target in zip(x_batch, y_batch):
            cell = cell.unsqueeze(0)
            output, hidden = model(cell, hidden) 
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
    val_loss = evaluate(model, criterion, val_loader)
    
    print(f'epoch: {epoch}, train loss: {train_loss}, val_loss : {val_loss}')


# test
test_loss = evaluate(model, criterion, test_loader)
print(f"test loss : {test_loss}")
