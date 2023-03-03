import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

#%%
# 단순 선형 회귀

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

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
# 소프트맥스 회귀 클래스로 구현하기

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)


y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print(y_one_hot.shape)


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        #
        
    def forward(self, x):
        #
    
model = SoftmaxClassifierModel()    

# optimizer 설정


nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # y_pred 계산
    

    # loss 계산
    

    # loss로 y_pred 개선
   
   
   
   
    # 100번마다 로그 출력
    
    
#%%    