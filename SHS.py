

#%%
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

from itertools import repeat

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


import warnings
warnings.filterwarnings('ignore')

#%%
from torch.utils.data import DataLoader, Dataset

#%%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#%%
# data generation
seed = 0
torch.manual_seed(seed)

mean_vec = list(repeat(1, 50)) + list(repeat(3, 50))
mean = torch.Tensor([mean_vec, list(reversed(mean_vec))])
'''
tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3., 3., 3., 3.,
         3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
         3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
         3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
         3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
         3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
'''
cov1 = torch.eye(100)
#torch.eye() : 대각선에 1이 있고 다른 곳에는 0으로 채워진, 크기가 nxn인 텐서
cov = torch.stack([cov1, cov1], 0)
#torch.stack() : 새로운 차원을 기준으로 두 텐서를 연결하며, 대상이 되는 텐서의 모양이 모두 같아야 한다
'''
dim = 0 → unsqueeze(0)으로 텐서를 확장한 뒤 dim = 0 으로 병합한 것과 동일 
dim = 1  → unsqueeze(1)로 텐서로 확장한 뒤 dim = 1 으로 병합한 것과 동일
tensor([[[1., 0., 0.,  ..., 0., 0., 0.],
         [0., 1., 0.,  ..., 0., 0., 0.],
         [0., 0., 1.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 1., 0., 0.],
         [0., 0., 0.,  ..., 0., 1., 0.],
         [0., 0., 0.,  ..., 0., 0., 1.]],

        [[1., 0., 0.,  ..., 0., 0., 0.],
         [0., 1., 0.,  ..., 0., 0., 0.],
         [0., 0., 1.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 1., 0., 0.],
         [0., 0., 0.,  ..., 0., 1., 0.],
         [0., 0., 0.,  ..., 0., 0., 1.]]])
'''

distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)
#MultivariateNormal(loc: torch.Size([2, 100]), covariance_matrix: torch.Size([2, 100, 100]))

x = distrib.rsample().T
#x.shape :torch.Size([100, 2]) 
beta = torch.rand(2).uniform_(-1, 1)
#rand() :  0과 1 사이의 숫자를 균등하게 생성
#tensor([ 0.8075, -0.8115])
y = torch.bernoulli(torch.sigmoid(x @ beta)).to(torch.float32)
#y = torch.tensor(list(repeat(1., 50)) + list(repeat(0., 50)))->0 또는 1
#y.shape : torch.Size([100])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = seed)

#%%

##데이터셋 만들기
class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index]
        
    def __len__(self): 
        return len(self.x_data)

#%%

train_set = MyDataset(x_train, y_train)
train_loader = DataLoader(train_set, batch_size = 10, shuffle = True)
test_set = MyDataset(x_test, y_test)
test_loader = DataLoader(test_set, batch_size = 1)

#%%
# 1. Logistic Regression 모형 작성
class LogisticRegression(nn.Module):
    def __init__(self,p):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(p, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

#%%
# 2. train function 코드 작성 
def train(model, device, criterion, loader):
    model.train()
    
    for step, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        target = target.unsqueeze(1)
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    

#%%
# 3. evaluation function 코드 작성
@torch.no_grad()
def eval(model, device, criterion, loader):
    model.eval()
    test_loss = 0
    correct = 0

    for step, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        pred_prob = model(data)
        target = target.unsqueeze(1)
        loss = criterion(pred_prob, target)
        test_loss += loss
        pred = pred_prob.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    test_loss /= len(loader.dataset)
    acc = correct/len(loader.dataset) * 100
    return test_loss, acc

#%%
# 4. 학습 코드 작성
input_dim = 2
model = LogisticRegression(input_dim) 
criterion = nn.BCELoss() #이진분류문제
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(10):
    train(model, device, criterion, train_loader)
    train_loss, train_acc = eval(model, device, criterion, train_loader)
    test_loss, test_acc = eval(model, device, criterion, test_loader)
    print(f'epoch: {epoch}, train loss: {train_loss}, test_loss: {test_loss}, test acc: {test_acc}%')


'''
train function, test function을 만드려고 이전 코드들을 참고해서 data loader 부분을 만들기 위해
dataset을 새롭게 만들었는데, criterian 적용 부분에서 input size는 torch.Size([10, 1]), target size는 torch.Size([10])로 다르게 나옵니다.
그래서 unsqueeze함수를 추가했는데, 차원을 마음대로 변경시켜도 문제가 없는지 궁금합니다.
그리고 차원을 변경시킬 때, target size를 변경시키는 게 좋을지, input size를 변경시키는 게 좋을지 궁금합니다!
'''
