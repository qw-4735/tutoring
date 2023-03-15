
#%%
import torch
import torch.nn as nn #뉴럴넷 구성 요소
import torch.nn.functional as F #딥러닝에 자주 사용되는 수학적 함수
import torch.optim as optim #최적화 함수
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms #torchvision : 딥 러닝에 사용되는 여러 데이터셋에 대한 모듈, datasets : 여러 데이터를 가지고 있다, transforms : 데이터의 형태 지정 가능
from matplotlib import pyplot as plt
from tqdm import tqdm

#%%
# 연산에 사용할 device 설정
#torch.cuda.is_available(): cuda가 사용 가능하면 true를 반환함으로서 device에 cuda를 설정하도록 한다
if torch.cuda.is_available():
    device = torch.device('cuda')
    #.device('cuda'): 모델을 GPU에 넣어준다
else:
    device = torch.device('cpu')

#%%
# load mnist
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
'''
transforms.ToTensor(): 이미지를 Tensor 형태로 변경
   -> pixels 값들을 [0~255]에서 [0.0~1.0]으로 자동 변환 
transforms.Normalize((0.1307,), (0.3081,)) 
- mean : (sequence)형식으로 평균을 입력하며, 괄호 안에 들어가 있는 수의 개수가 채널의 수이다
- std : (sequence)형식으로 표준을 입력하며, 마찬가지로 괄호 안에 들어가 있는 수의 개수가 채널의 수이다.
- transofrms.Normalize((R채널 평균, G채널 평균, B채널 평균), (R채널 표준편차, G채널 표준편차, B채널 표준편차))로 입력하여 적용할 수 있다
  (https://teddylee777.github.io/pytorch/torchvision-transform/)
'''

'''
Q1. normalization과정에서 평균은 (0.1307,), stds는 (0.3081,)로 설정했는데 mnist는 2개의 채널 중 한 채널만 normalize했다고 보면 되는 걸까요?
'''

train_dataset = datasets.MNIST('./', train = True, download = True, transform = transform)
test_dataset = datasets.MNIST('./', train = False, download = True, transform = transform)
'''
train: train or test 데이터를 받아온다->True를 주면 훈련 데이터를 리턴, False를 주면 테스트 데이터를 리턴
transform: 사전에 설정해 놓은 데이터 처리 형태->현재 데이터를 파이토치 텐서로 변환
download: 해당 경로에 MNIST 데이터가 없다면 다운로드한다
'''

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = False)
'''
train_loader : MNIST의 train 데이터를 받아오는 함수
batch_size : batch 단위 만큼 데이터를 뽑아온다
shuffle : 데이터를 shuffle할 것인지 여부->true : 데이터의 순서를 학습하지 못하게 하는 것이다
''' 

#%%
# 데이터 구조 확인
# 하나의 데이터만 확인해보면 tuple형태로 되어있고 첫 번째 원소는 이미지, 두 번째 원소는 이미지의 true label인 것을 알 수 있음
a = train_dataset[0]
print(a)
plt.imshow(a[0].squeeze(0))
'''
squeeze() : 차원이 1인 차원을 제거한다.
pytorch에서 불러오는 MNIST 데이터의 경우 [1, 28, 28]로 구성된다.
따라서 1을 없애야 이미지를 그릴 수 있다
(https://dreamofadd.tistory.com/112 참조)
'''

#%%
# model
class CNN(nn.Module):
#class를 이용해 nn.Module 상속
    def __init__(self):
        super(CNN, self).__init__()
        #super(): 상속받은 부모 클래스-> nn.Module 클래스의 속성들을 가지고 초기화 
        #__init__() : 부모클래스의 생성자를 불러주는 것이다->파이썬에서 객체가 갖는 속성값을 초기화하는 역할로, 객체가 생성될 때 자동으로 호출
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #첫번째 합성곱 층
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #두번째 합성곱 층
        '''
        Conv2d (이미지 분류에서 많이 사용)
        Conv2d(in_channels, out_channels, kernel_size, stride)
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        kernel_size: 커널 사이즈
        stride: stride 사이즈
        '''
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        #nn.Dropout : 특정 확률에 따라 dropout
        
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        #nn.Linear(input_dim, output_dim) : 선형회귀모델
        

    #foward() 함수: 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 함수
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = F.max_pool2d(x, 2)
        #max_pool2d : 컨볼루션 레이어를 지나고나서 풀링작업을 진행할때 쓰는 함수
        #첫번째는 input에 대한 데이터, 두번째는 풀링윈도우의 사이즈 정의
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        #torch.flatten(x, 1) : 배치 차원을 제외한 모든 차원을 하나로 평탄화
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x

'''
Q2 dropout 위치, relu를 쓰는 주기 등을 따로 선정하는 기준이 있는지 궁금합니다!
'''

def train(model, device, criterion, loader):
    model.train()
    #모듈을 평가 모드로 설정한다
    
    for step, (data, target) in enumerate(loader):
        #enumerate() : 리스트의 원소에 순서값을 부여해주는 함수    
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #.zero_grad() : 갱신할 Variable들에 대한 모든 변화도를 0으로 만든다
        pred = model(data)
        #모델에 data를 전달하여 예상하는 target 값을 계산한다
        loss = criterion(pred, target)
        #모델에서 나온 pred_prob과 target을 이용해 loss를 계산한다
        loss.backward()
        #.backward() : 역전파 단계 실행. 모델의 Variable들에 대한 손실의 변화도를 계산한다
        optimizer.step()
        #.step() : 가중치를 갱신한다
    

@torch.no_grad()
def eval(model, device, criterion, loader):
    model.eval()
    #모델 내부의 모든 layer가 evaluation 모드가 된다
    test_loss = 0
    correct = 0
    #테스트 오차와 예측이 맞은 수를 담을 변수를 0으로 초기화한다
    
    for step, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        pred_prob = model(data)
        #모델에 data를 전달하여 예상하는 target 값을 계산한다
        loss = criterion(pred_prob, target)
        #모델에서 나온 pred_prob과 target을 이용해 loss 계산한다
        test_loss += loss
        #모든 오차를 더한다
        pred = pred_prob.argmax(dim=1, keepdim=True)
        '''
        torch.argmax(input, dim, keepdim):가장 큰 인덱스를 구한다
        input: Tensor
        dim: 몇번째 축을 기준으로 argmax연산을 할지 결정한다
        keepdim:argmax연산을 한 축을 생략할지 그대로 둘 지 결정한다
        (https://gaussian37.github.io/dl-pytorch-snippets/)
        '''
        correct += pred.eq(target.view_as(pred)).sum().item()
        '''
        예측과 정답을 비교하여 일치할 경우 correct에 1을 더한다
        eq() : 추출한 모델의 예측과 레이블이 일치하는지를 알아본다-> 값이 일치하면 1, 아니면 0을 출력한다
        view_as() : target 텐서를 인수(pred)의 모양대로 다시 정렬한다
        sum() : 맞은 것인 1들의 합이 구해진다
        item() : 딕셔너리 값들을 쌍으로 불러낸다
        '''
        
    test_loss /= len(loader.dataset)
    #test_loss의 평균을 구한다
    acc = correct/len(loader.dataset) * 100
    #정답 평균에는 100을 곱하여 정확도를 구한다 
    return test_loss, acc
    
    
#%%
model = CNN().to(device)
## CNN 모델을 정의한다
criterion = nn.CrossEntropyLoss()
#손실함수 지정한다 / nn.CrossEntropyLoss()의 경우 기본적으로 LogSoftmax()가 내장
optimizer = optim.Adam(model.parameters(), lr=0.001)
''' 
optimizer : 최적화 함수 지정한다 -> Adam 
model.parameters(): model의 파라미터들을 할당한다
lr : learning_rate 지정한다
'''

#%%
#10번 학습 진행
for epoch in tqdm(range(10)):
    train(model, device, criterion, train_loader)
    train_loss, train_acc = eval(model, device, criterion, train_loader)
    test_loss, test_acc = eval(model, device, criterion, test_loader)
    print(f'epoch: {epoch}, train loss: {train_loss}, test_loss: {test_loss}, test acc: {test_acc}%')


'''
Q3 한 번 학습 진행하는데 5분 넘게 걸리는데 이렇게 느린 게 정상인지 궁금합니다..!
'''