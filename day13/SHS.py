'''
seq2seq는 인코더와 디코더로 구성
인코더는 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤 마지막에 이 모든 단어 정보들을 압축해서 하나의 벡터인 컨텍스트 벡터 생성
인코더는 컨텍스트 벡터(인코더 RNN 셀의 마지막 시점의 은닉 상태)를 디코더(디코더 RNN 셀의 첫번째 은닉 상태)로 전송

train 과정 : 정답을 알려주면서 훈련
test 과정 : 디코더는 컨텍스트 벡터와 <sos>만을 입력으로 받은 후에 다음에 올 단어를 예측하고, 그 단어를 다음 시점의 RNN 셀의 입력으로 넣는 행위를 반복
'''



#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import trange
import random
import torch.optim as optim

#%%
data = pd.read_csv("서인천IC-부평IC 평균속도.csv", encoding='CP949')
plt.figure(figsize=(20,5))
plt.plot(range(len(data)), data["평균속도"])
data.head() #집계일시, 평균속도로 이루어짐
#집계일시는 2021050100~	2021053123 1시간 단위

#Data Preprocessing
#데이터분리->스케일링->슬라이딩->텐서변환->dataloader
#data["평균속도"] = min_max_scaler.fit_transform(data["평균속도"].to_numpy().reshape(-1,1))

train = data[:-24*7]
train = train["평균속도"].to_numpy()

train.size #576

test = data[-24*7:]
test = test["평균속도"].to_numpy()

test.size #168

data.shape #(744, 3)

'''
이번 과제에서는 24*14로 24*7을 예측하는데, 저번 과제에서처럼 하려면
test에서 -(24*7+24*14)로 나타내어야할 것 같은데 data에서 앞부분의 데이터 수가 더 적으면 어떻게 처리해아할까요?
'''


#일주일의 데이터를 예측

#%%
#스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.reshape(-1,1))
test_scaled = scaler.transform(test.reshape(-1,1))

train_scaled.shape #(576, 1)

#%%
#Sliding Window Dataset
from torch.utils.data import DataLoader, Dataset

#num_samples = (train_scaled.shape[0] - 24*14-24*7)//1 +1
X0 = np.zeros([24*14,73])
X0.shape
X0[:24*14,1:2].shape
#X1= X0.reshape(X0.shape[0], X0.shape[1], 1)
#X1.shape
#X1.transpose([1,0,2]).shape #(73, 336, 1)

class windowDataset(Dataset):
    def __init__(self, y, input_window, output_window, stride=1):
        L = y.shape[0] #총 데이터의 개수(576개)
        
        #stride씩 움직일 때 생기는 총 sample의 개수(73개)
        num_samples = (L - input_window - output_window) // stride + 1

        #input과 output : shape = (window 크기, sample 개수)
        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i:i+1] = y[start_x:end_x] #i번째 sample의 X
            #broadcast 오류 X[:,i]->X[:,i:i+1]
            
            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i:i+1] = y[start_y:end_y] #i번째 sample의 Y

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        #(336, 73) -> (336, 73, 1) 
        #X.shape[0]:336, X.shape[1]:73
        #transpose((1,0,2)) -> (73, 336, 1) 
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        #(168, 73) -> (168, 73, 1) 
        #Y.shape[0]:336, Y.shape[1]:73 
        #transpose((1,0,2))->(73,168,1)
        
        self.x = X
        self.y = Y
        
        self.len = len(X)
    
    #결과의 첫 번째 값으로는 input, 두 번째 값으로는 output이 출력
    def __getitem__(self, i):
        return self.x[i], self.y[i] 
    
    def __len__(self):
        return self.len

iw = 24*14
ow = 24*7

train_dataset = windowDataset(train_scaled, input_window=iw, output_window=ow, stride=1)
# input_window : 24*14 로 output_window : 24*7 예측
# stride=1 -> 1시간씩 이동

train_loader = DataLoader(train_dataset, batch_size=64)

for  data1 in train_loader:
      print("Data: ", data1[0].shape, ", y: ",data1[1].shape)
      break
#Data:torch.Size([64, 336, 1]), y:torch.Size([64, 168, 1])

#%%
#Modeling

#LSTM encoder
#input 으로부터 입력을 받고 lstm을 이용하여 디코더에 전달할 hidden state를 생성
class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        return lstm_out, self.hidden

#%%

next(iter(train_loader))[0].shape #torch.Size([64, 336, 1])
next(iter(train_loader))[1].shape #torch.Size([64, 168, 1])

a=lstm_encoder(input_size = 1, hidden_size = 16)    
for x in train_loader:
    print("x[0].shape =", x[0].shape)
    output, hidden = a(x[0].float())
    #print("x[0].shape =",output.shape)
    #hidden은 (hn,cn) tuple 형태

'''
h_out(hidden cell output)은 시간 단계 수와 관련하여 마지막 hidden state를 나타냄
num_layer>1인 경우 모든 레이어에 대한 숨겨진 상태가 포함
'''
'''
x[0].shape = torch.Size([64, 336, 1])
torch.Size([64, 336, 16])
x[0].shape = torch.Size([9, 336, 1])
torch.Size([9, 336, 16])

위에 next(iter(train_loader))[0].shape 에서는 torch.Size([64, 336, 1])가 나왔는데,
아래 train_loader에서 torch.Size([9, 336, 1])도 포함되어서 나왔는데, 잘못 설계한 걸까요?
'''    

#%%
#LSTM decoder   
#sequence의 이전값 하나와, 이전 결과의 hidden state를 입력 받아서 다음 값 하나를 예측
class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,num_layers = num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)           

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(-1), encoder_hidden_states)
        #unsqueeze(-1) : 마지막 차원에 1인 차원을 추가
                
        output = self.linear(lstm_out)
        
        return output, self.hidden

#%%
b=lstm_decoder(input_size = 1, hidden_size = 16) 


'''
Inputs: input, (h_0, c_0)
input - (L, H_in) : unbatched input
      - (L, N, H_in) : batch_first=False
      - (N, L, H_in) : batch_first=True

(L = sequence length, N : batch size, H_in : input_size) 

h_0 - (D*num_layers, H_out) : unbatched input
    - (D*num_layers, N, H_out) : containing the initial cell state for each element in the input sequence
default to zeros
c_0 - (D*num_layers, H_cell) : unbatched input
    - (D*num_layers, N, H_cell) : containing the initial cell state for each element in the input sequence

(h_cell = hidden_size)

output - (L, D*H_out) : unbatched input
       - (L, N, D*H_out) : batch_first=False
       - (N, L, D*H_out) : batch_first=True
'''
'''
rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
output.shape # torch.Size([5, 3, 20])
hn.shape # torch.Size([2, 3, 20])
cn.shape # torch.Size([2, 3, 20])
'''

outputs = torch.zeros(64, 24*7, 1)
#output : (N=64, L=24*7, H_in=1)


#%%
class lstm_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio):
        batch_size = inputs.shape[0] #64
        input_size = inputs.shape[2] #1

        outputs = torch.zeros(batch_size, target_len, input_size)
        #output : (N=64, L=24*7, H_in=1)
        
        _, hidden = self.encoder(inputs) #encoder에서 나온 hidden 이용
        decoder_input = inputs[:,-1, :]  #torch.Size([64, 1])  torch.Size([9, 1])
        
        #원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len): 
            out, hidden = self.decoder(decoder_input, hidden)
            out =  out.squeeze(1)
            #squeeze(1):차원이 1인 경우에는 해당 차원을 제거
            
            # teacher forcing을 구현한다.
            # teacher forcing에 해당하면 다음 인풋값으로는 예측한 값이 아니라 실제 값을 사용한다.
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :] #torch.Size([64, 1]) torch.Size([9, 1])
                #디코더의 인풋으로 실제 값을 넣어줌
            else:
                decoder_input = out
            outputs[:,t,:] = out

        return outputs
	
    # 편의성을 위해 예측해주는 함수도 생성한다.
    def predict(self, inputs, target_len):
        self.eval()
        inputs = inputs.unsqueeze(0)
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size)
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:,-1, :]
        for t in range(target_len): 
            out, hidden = self.decoder(decoder_input, hidden)
            out =  out.squeeze(1)
            decoder_input = out
            outputs[:,t,:] = out
        return outputs.detach().numpy()[0,:,0]

#for x,y in train_loader : 
#    print(y[:, 2, :].shape) #torch.Size([64, 1]) torch.Size([9, 1])

'''
0번째 hidden state와 그 가중치의 곱 + 1번째 Input값과 그 가중치의 곱 = 1번째 hidden state


output 추출 방법
1. 각 time step마다 output이 필요한 경우 
(마지막 LSTM cell의 hidden_state 추출)
last_output = hidden[0].squeeze()
print(last_output)

2. 마지막 LSTM cell의 output만 필요한 경우
(out의 마지막 열들을 모두 추출)
last_output = out[:,-1]
print(last_output)
'''  

#%%
#Train
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = lstm_encoder_decoder(input_size=1, hidden_size=16).to(device)
#input feature : 속도(1개), hidden_size =16

learning_rate=0.01
epoch = 100 #epoch 3000->100으로 수정
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

from tqdm import tqdm


model.train()
with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        for x,y in train_loader:
            optimizer.zero_grad()
            #Data:torch.Size([64, 336, 1]), y:torch.Size([64, 168, 1])
            x = x.to(device).float()
            #nn.LSTM은 float형을 기반으로 연산
            y = y.to(device).float()
            output = model(x, y, ow, 0.6).to(device)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()
        tr.set_postfix(loss="{0:.5f}".format(total_loss/len(train_loader)))
        

#Evaluate
#https://doheon.github.io/%EC%BD%94%EB%93%9C%EA%B5%AC%ED%98%84/time-series/ci-3.lstm-post/
predict = model.predict(torch.tensor(train[-24*7*2:]).reshape(-1,1).to(device).float(), target_len=ow)
real = data["평균속도"].to_numpy()

predict = scaler.inverse_transform(predict.reshape(-1,1))
real = scaler.inverse_transform(real.reshape(-1,1))

#%%
plt.figure(figsize=(20,5))
plt.plot(range(400,744), real[400:], label="real")
plt.plot(range(744-24*7,744), predict[-24*7:], label="predict")

plt.title("Test Set")
plt.legend()
plt.show()


def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPEval(predict[-24*7:],real[-24*7:]) #98.60644993301997
# %%
