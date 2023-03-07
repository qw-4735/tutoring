import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
  
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
        self.linear = nn.Linear(4, 3)
        #4개의 특성을 가지고 3개의 클래스로 분류
        
    def forward(self, x):
        return self.linear(x)
    
model = SoftmaxClassifierModel()    

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # y_pred 계산
    y_pred = model(x_train)

    # loss 계산
    loss = F.cross_entropy(y_pred, y_train)

    # loss로 y_pred 개선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
   
    # 100번마다 로그 출력
     if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
    
#%%    
