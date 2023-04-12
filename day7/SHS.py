# 6일차에서 작성한 RNN 모형을 이용하여 시간당 에너지 소비량 (시계열 데이터) 예측 코드 작성

# https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b 
# data : https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
# fitting the RNN model constructed in day6 (모델 구현할 때 line by line으로 차원 계산해보고 주석 달 것)

#%%
#EDA
import plotly.graph_objs as go #go를 통해서 Figure객체를 선언하고 Figure내에 필요한 Data와 Layout등을 설정해주는 방식
from plotly.offline import iplot

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

#%%
import pandas as pd

df = pd.read_csv('AEP_hourly.csv')
print(df)

df = df.set_index(['Datetime'])
df.index = pd.to_datetime(df.index)
if not df.index.is_monotonic:
    df = df.sort_index()
    
df = df.rename(columns={'AEP_MW': 'value'})
df.shape
plot_dataset(df, title='PJM East (AEP_MW) Region: estimated energy consumption in Megawatts (MW)')

#%%
#X(t+n)를 예측하기 위한 time_lags 생성
def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    #shift(n): 행의 위치를 일정 칸수씩 이동킨다
    #1.데이터프레임의 행을 위,아래로 옮기고 싶을때. 2. 데이터의 변화량을 저장하는 컬럼을 만들고 싶을 때
    df_n = df_n.iloc[n_lags:]
    return df_n
    
input_dim = 100

df_generated = generate_time_lags(df, input_dim)
df_generated.shape

#%%
#DateTime 인덱스에서 날짜 시간 특성을 만들기
df_features = (
                df
                .assign(hour = df.index.hour)
                .assign(day = df.index.day)
                .assign(month = df.index.month)
                .assign(day_of_week = df.index.dayofweek)
                .assign(week_of_year = df.index.week)
              )


df_features.shape
#%%
'''
def onehot_encode_pd(df, col_name):
    dummies = pd.get_dummies(df[col_name], prefix=col_name)
    return pd.concat([df, dummies], axis=1).drop(columns=[col_name])

df_features = onehot_encode_pd(df_features, ['month','day','day_of_week','week_of_year'])

위의 코드가 ValueError: Length of 'prefix' (4) did not match the length of the columns being encoded (0)
라는 오류가 떠서 아래 코드로 바꿔서 실행했을 때에는 실행이 되었는데, 어떤 부분이 잘못되었을까요?
'''
#%%
def onehot_encode_pd(df, col_name):
    dummies = pd.get_dummies(df,columns=col_name, prefix=col_name) #get_dummies:0과 1로만 이루어진 열을 생성->주어진 데이터 세트에서 원-핫 인코딩 열을 빠르게 생성
    return dummies

df_features_onehot = onehot_encode_pd(df_features, ['month','day','day_of_week','week_of_year'])
df_features_onehot

#%%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def onehot_encode(df, onehot_columns):
    ct = ColumnTransformer(
        [('onehot', OneHotEncoder(drop='first'), onehot_columns)],
        remainder='passthrough'
        )
    #ColumnTransformer:숫자형과 범주형 변수 전처리 파이프라인 합치기; 입력 데이터에 있는 열마다 다른 변환을 적용
    #OneHotEncoder:각 범주별로 칼럼을 만들어서 해당 범주에 속하면 '1' (hot), 해당 범주에 속하지 않으면 '0' (cold) 으로 인코딩
    return ct.fit_transform(df)

onehot_columns = ['hour']
onehot_encoded = onehot_encode(df_features, onehot_columns)

onehot_encoded

'''
 원-핫 인코딩은 DateTime 기능의 순환 패턴을 완전히 캡처하지 못하기 때문에 다른 방법 사용
'''
#%%
import numpy as np

#시간의 원래 값을 사용하는 대신 주기성을 유지하면서 사인 변환을 사용
def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
             }
    return df.assign(**kwargs).drop(columns=[col_name])
#assign(): DataFrame에 새 열을 할당하는 메서드
#kwargs : 새열이름 = 내용 형식으로 입력되는 키워드

df_cyclical_features = generate_cyclical_features(df_features_onehot, 'hour', 24, 0)
# df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 0)
# df_features = generate_cyclical_features(df_features, 'month', 12, 1)
# df_features = generate_cyclical_features(df_features, 'week_of_year', 52, 0)

df_cyclical_features

#%%
from datetime import date
import holidays

# 1년 중 휴일이 에너지 소비 패턴에 영향을 미치는지 확인

us_holidays = holidays.US() #공휴일 확인

#주어진 날짜가 실제로 휴일인지 여부를 나타내는 이진 값이 있는 추가 열을 생성
def is_holiday(date):
    date = date.replace(hour = 0)
    return 1 if (date in us_holidays) else 0

def add_holiday_col(df, holidays):
    return df.assign(is_holiday = df.index.to_series().apply(is_holiday))


df_holiday_features = add_holiday_col(df_cyclical_features, us_holidays)
df_holiday_features[df_holiday_features['is_holiday']==1]


#%%
from sklearn.model_selection import train_test_split

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    #shuffle=false : 세트로 분할하는 동안 섞는 것을 방지
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_holiday_features, 'value', 0.2)

#%%
#Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
#X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
X_train_arr = scaler.fit_transform(X_train)
X_val_arr = scaler.transform(X_val)
X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
y_test_arr = scaler.transform(y_test)

#%%
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()
    
scaler = get_scaler('robust')

#%%
#데이터세트를 DataLoaders에 로드
from torch.utils.data import TensorDataset, DataLoader
import torch

batch_size = 64

train_features = torch.Tensor(X_train_arr)
train_targets = torch.Tensor(y_train_arr)
val_features = torch.Tensor(X_val_arr)
val_targets = torch.Tensor(y_val_arr)
test_features = torch.Tensor(X_test_arr)
test_targets = torch.Tensor(y_test_arr)

train_features[0].shape
test_targets[0].shape

train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

#%%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


#%%
#Training the model
from datetime import datetime
import matplotlib.pyplot as plt
class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()
        hidden = self.model.initHidden()
        # Makes predictions
        yhat, hidden = self.model(x, hidden)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        model_path = f'models/{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                #사이즈가 -1로 설정되면 다른 차원으로부터 해당 값을 유추
                
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)   
    
    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                #detach():gradient의 전파를 멈추는 역할
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()
        
#%%
#네트워크 생성
import torch.nn as nn

'''
#첫번째 (combined = torch.cat((input, hidden), 1) error)
#Tensors must have same number of dimensions: got 3 and 2

#RNN 모듈은 입력 및 은닉 상태로 작동하는 2개의 선형 계층이며, 출력 다음에 LogSoftmax 계층
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
'''
#%%
'''
#두번째
#Sizes of tensors must match except in dimension 1. Expected size 64 but got size 3 for tensor number 1 in the list.
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    #initHidden : 최초 RNN이 호출을 위해 state가 없을때 hidden vector를 생성
    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)        
'''       
#%%
'''
첫번재 시도에서 3개의 dimension이 필요하다고 해서 rnn모듈을 쓰지 않고 layer dim을 추가해서 모델을 만드려고 하는데, 
concat 함수에서 차원이 맞지 않는 오류가 나타나는데 어떻게 해결할 수 있을까요?

'''
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #input=input.view(self.num_layers,-1,self.hidden_size)->shape '[3, -1, 64]' is invalid for input of size 6784
        combined = torch.cat((input, hidden), 1)
        '''
        cat:원하는 dimension 방향으로 텐서를 나란하게 쌓아준다
        x = torch.rand(batch_size, N, K) # [M, N, K]
        y = torch.rand(batch_size, N, K) # [M, N, K]
        torch.cat([x,y], dim=1) #[M, N+N, K]
        '''
                
        ## 이 경우 hidden cell을 반복적으로 입력받도록 구현해야 하므로  hidden또한 return값에 포함해야 한다.
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output_softmax = self.softmax(output)
        return output_softmax, hidden

    #initHidden : 최초 RNN이 호출을 위해 state가 없을때 hidden vector를 생성
    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)        
    
    #output : [batch_size, sequence, hidden_size]
    #h_0:(D∗num_layers,N(batch_size),H out(hidden_size))


a=torch.zeros(3,1,64)
a.shape #torch.Size([3, 1, 64])
x_batch=train[0][0]
x_batch = x_batch.view([1, -1, len(X_train.columns)])
x_batch.shape #torch.Size([1, 1, 106])



#%%
#training
import torch.optim as optim
input_dim = len(X_train.columns)
input_dim #106
output_dim = 1
hidden_dim = 64
learning_rate = 1e-3
weight_decay = 1e-6
n_epochs = 100
layer_dim = 3
model = RNN(input_dim, hidden_dim, output_dim, layer_dim)
loss_fn = nn.MSELoss(reduction="mean")
#mse로 loss 설정
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)

opt.plot_losses()

predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)


#%%
#다차원 텐서를 1차원 벡터로 줄이기
def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
        #inverse_transform:MinMaxScaler로 표준화된 데이터를 다시 원본 데이터로 복원
    return df


def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    #np.ravel(x, order='C') : C와 같은 순서로 인덱싱하여 평평하게 배열 
    
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


df_result = format_predictions(predictions, values, X_test, scaler)

#%%
#오차 지표 계산
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(df):
    return {'mae' : mean_absolute_error(df.value, df.prediction),
            'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
            'r2' : r2_score(df.value, df.prediction)}

result_metrics = calculate_metrics(df_result)

#%%
#baseline 예측 생성 - 선형 회귀
from sklearn.linear_model import LinearRegression

def build_baseline_model(df, test_ratio, target_col):
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    result = pd.DataFrame(y_test)
    result["prediction"] = prediction
    result = result.sort_index()

    return result

df_baseline = build_baseline_model(df_features, 0.2, 'value')
baseline_metrics = calculate_metrics(df_baseline)

#%%
#예측 시각화
import plotly.offline as pyo

def plot_predictions(df_result, df_baseline):
    data = []
    
    value = go.Scatter(
        x=df_result.index,
        y=df_result.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df_result.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    baseline = go.Scatter(
        x=df_baseline.index,
        y=df_baseline.prediction,
        mode="lines",
        line={"dash": "dot"},
        name='linear regression',
        marker=dict(),
        text=df_baseline.index,
        opacity=0.8,
    )
    data.append(baseline)
    
    prediction = go.Scatter(
        x=df_result.index,
        y=df_result.prediction,
        mode="lines",
        line={"dash": "dot"},
        name='predictions',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction)
    
    layout = dict(
        title="Predictions vs Actual Values for the dataset",
        xaxis=dict(title="Time", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)
    
    
# Set notebook mode to work in offline
pyo.init_notebook_mode()

plot_predictions(df_result, df_baseline)
