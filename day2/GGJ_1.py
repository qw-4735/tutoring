import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
#GPU를 통해 만들어내는 결과들에 randomness 를 통제

#%%
# 단순 선형 회귀

#훈련데이터인 x_train과 y_train을 선언
x_train = torch.FloatTensor([[1], [2], [3]])
#(3×1)의 크기를 가지는 2차원 텐서(행렬) 생성
y_train = torch.FloatTensor([[2], [4], [6]])
#(3×1)의 크기를 가지는 2차원 텐서(행렬) 생성


W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 2000 
for epoch in range(nb_epochs + 1):

    
    y_pred = x_train * W + b

    loss = torch.mean((y_pred- y_train) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
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

    def forward(self, x):
        return self.linear(x)
    
model = LinearRegressionModel()    

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

nb_epochs = 2000
for epoch in range(nb_epochs+1):

    y_pred = model(x_train)

    loss = F.mse_loss(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
 
    if epoch % 100 == 0:
      print('Epoch {:4d}/{} Loss: {:.6f}'.format(
          epoch, nb_epochs, loss.item()
      ))

#%%
