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