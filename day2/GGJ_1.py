
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
  
torch.manual_seed(1)
#GPU를 통해 만들어내는 결과들의 randomness 를 통제

#%%
# 단순 선형 회귀

#훈련데이터인 x_train과 y_train을 선언
x_train = torch.FloatTensor([[1], [2], [3]])
#(3×1)의 크기를 가지는 2차원 텐서(행렬) 생성
y_train = torch.FloatTensor([[2], [4], [6]])
#(3×1)의 크기를 가지는 2차원 텐서(행렬) 생성


W = torch.zeros(1, requires_grad=True)
#zeros : W를 0으로 초기화
#requires_grad=True : 해당 텐서에 대한 계산 모두 tracking해서 기울기 구해주기->값이 변경된다
b = torch.zeros(1, requires_grad=True)
#b를 0으로 초기화

optimizer = optim.SGD([W, b], lr=0.01)
#경사 하강법 구현
#W,b는 parameter
#lr은 Learning Rate의 줄임말이며, 미분값을 얼만큼 이동시킬 것인가를 결정

nb_epochs = 2000 
#학습 주기 = 2000

for epoch in range(nb_epochs + 1):

    
    y_pred = x_train * W + b
    
    loss = torch.mean((y_pred- y_train) ** 2)
    #loss는 평균제곱오차이다
    optimizer.zero_grad()
    #zero_grad():기울기를 0으로 초기화
    loss.backward()
    #backward():W와 b에 대한 역전파 시행
    optimizer.step()
    #step():각 layer의 파라미터와 같이 저장된 gradient 값을 이용하여 파라미터를 업데이트

    if epoch % 100 == 0:
        #100번마다 표시
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} loss: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), loss.item()
        ))

#%%        
# 단순선형회귀 클래스로 구현하기

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        #모델 선언, 단순 선형 회귀이므로 input_dim=1, output_dim=1

    def forward(self, x):
        return self.linear(x)
    
model = LinearRegressionModel()    

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    y_pred = model(x_train)

    loss = F.mse_loss(y_pred, y_train)
    #loss는 파이토치에서 제공하는 평균제곱오차이다

    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
 
    if epoch % 100 == 0:
      print('Epoch {:4d}/{} Loss: {:.6f}'.format(
          epoch, nb_epochs, loss.item()
      ))

#%%


'''
(3×1)의 크기를 가지는 2차원 텐서가 어떤 구조인지 잘 이해가 안 되는 것 같습니다.
또, 다른 예제들을 찾아봤을 때, model = nn.Linear(1,1)으로 모델을 선언하는데
class로 모델을 따로 만드는 이유가 있는지 궁금합니다!

'''
