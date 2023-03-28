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


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

#%%
# data generation
seed = 0
torch.manual_seed(seed)

mean_vec = list(repeat(1, 50)) + list(repeat(3, 50))
mean = torch.Tensor([mean_vec, list(reversed(mean_vec))])
cov1 = torch.eye(100)
cov = torch.stack([cov1, cov1], 0)
distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)

x = distrib.rsample().T
#x -> torch.Size([100, 2])
beta = torch.rand(2).uniform_(-1, 1)
y = torch.bernoulli(torch.sigmoid(x @ beta)).to(torch.float32)
# y = torch.tensor(list(repeat(1., 50)) + list(repeat(0., 50)))
# y -> torch.Size([100])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = seed)
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)


#%%
# 1. Logit layer 작성
class Logit(torch.nn.Module):
    def __init__(self, dim):
        super(Logit, self).__init__()
        self.dim = dim
        self.fc = nn.Linear(self.dim, 1)
    def forward(self, x):
        return self.fc(x)


#%%
# 2. cross entropy 함수 작성
def Cross_entropy(output, target):
    loss = (-1.0)*torch.sum(torch.log(torch.exp(output)/torch.sum(torch.exp(output)))*target)
    return loss

#%%
# 3. Logit layer를 이용한 LogisticRegression 모형 작성
class LogisticRegression(nn.Module):
    def __init__(self,p):
        super(LogisticRegression, self).__init__()
        self.p = p
        self.sigmoid = nn.Sigmoid()
        self.logit = Logit(p)

    def forward(self, x):
        x = self.logit(x)
        x = self.sigmoid(x)
        return x

#%%
# 4. 학습 코드 작성
input_dim = 2
model = LogisticRegression(input_dim)
model_param_group = []
model_param_group.append({'params': model.parameters()})
optimizer = optim.SGD(model_param_group, lr = 0.005)

y_pred = model(x_train)
print(y_pred)

#%%
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    y_pred = model(x_train)
    loss = Cross_entropy(y_pred, y_train)
    #loss는 파이토치에서 제공하는 평균 제곱 오차 함수

    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
 
    if epoch % 100 == 0:
      print('Epoch {:4d}/{} Loss: {:.6f}'.format(
          epoch, nb_epochs, loss.item()
      ))


# %%
