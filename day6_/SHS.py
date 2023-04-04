# 1. https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial  
     #한 줄씩 실행해보고, 주석 꼼꼼히 달기

#%%
#데이터 준비
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)
#glob() : 인자로 받은 패턴과 이름이 일치하는 모든 파일과 디렉터리의 리스트를 반환

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
#영어 알파벳 소문자, 대문자 모두를 출력

n_letters = len(all_letters)

# 유니코드 문자열을 ASCII로 변환, https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        #join():매개변수로 들어온 리스트에 있는 요소 하나하나를 합쳐서 하나의 문자열로 바꾸어 반환하는 함수
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
        #category():문자의 유형을 설명
        #'Mn'은 "Mark, nonspacing"
    )
''' 
유니코드 정규화
NFC: Normalization Form Canonical Composition
NFD: Normalization Form Canonical Decomposition
NFKC: Normalization Form Compatibility Composition
NFKD: Normalization Form Compatibility Decompositio
'''

print(unicodeToAscii('Ślusàrski'))

# 각 언어의 이름 목록인 category_lines 사전 생성
# category(언어)를 line(이름)에 매핑
category_lines = {}
 
all_categories = []

# 파일을 읽고 줄 단위로 분리
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
print(n_categories)

print(category_lines['Italian'][:5])
#['Abandonato', 'Abatangelo', 'Abatantuono', 'Abate', 'Abategiovanni']

#%%
#이름을 Tensor로 변경
import torch

# all_letters 로 문자의 주소 찾기, 예시 "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)
    #find():주어진 문자열에서 하위 문자열이 처음 나타나는 인덱스를 찾기 


# 검증을 위해서 한 개의 문자를 <1 x n_letters> Tensor로 변환
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    #크기가 1 x n_letters인 one-hot vector
    tensor[0][letterToIndex(letter)] = 1
    
    return tensor

# 한 줄(이름)을  <line_length x 1 x n_letters>,
# 또는 One-Hot 문자 벡터의 Array로 변경
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))
'''tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0.]])
'''

print(lineToTensor('Jones').size())
#torch.Size([5, 1, 57])

#%%
#네트워크 생성
import torch.nn as nn

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

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

#--------------------------------------------
input = letterToTensor('A')
#입력(현재 문자 Tensor)

hidden = torch.zeros(1, n_hidden)
#이전의 은닉 상태(처음에는 0으로 초기화)를 전달

output, next_hidden = rnn(input, hidden)
#출력(각 언어의 확률)과 다음 은닉 상태(다음 단계를 위해 유지)를 돌려받음
#---------------------------------------------

input = lineToTensor('Albert')
print(input)
'''tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]],

        [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]]])
'''

hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)
'''
tensor([[-2.9514, -3.0068, -2.9876, -2.8444, -2.8842, -2.9764, -2.8821, -2.8204,
         -2.8186, -3.0446, -2.7953, -2.8877, -2.8141, -2.8922, -2.9073, -2.8373,
         -2.8538, -2.8667]], grad_fn=<LogSoftmaxBackward0>)
'''
#출력:1 x n_categories Tensor

#%%
#학습
#학습준비

#네트워크 출력(각 카테고리의 우도)으로 가장 확률이 높은 카테고리 이름(언어)과 카테고리 번호를 반환
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)  # 텐서의 가장 큰 값 및 주소
    category_i = top_i[0].item()   # 텐서에서 정수 값으로 변경
    return all_categories[category_i], category_i

print(categoryFromOutput(output))
#('Greek', 7)

#%%
#학습예시
import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]
    #random.randint(start, stop) : start 이상 stop 이하 범위의 정수 난수 생성

#학습 예시(하나의 이름과 그 언어)를 얻는 방법
def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
    print(category_tensor,'\n', line_tensor)
'''
category = German / line = Silverstein
category = Japanese / line = Kimiyama
category = Dutch / line = Hout
category = Greek / line = Vassilopulos
category = Portuguese / line = Almeida
category = French / line = Forestier
category = Russian / line = Vaidanovich
category = Arabic / line = Morcos
category = English / line = Murrell
category = Czech / line = Dopita
'''
#%%
#네트워크 학습
criterion = nn.NLLLoss()
#RNN의 마지막 계층이 nn.LogSoftmax ->nn.NLLLoss

learning_rate = 0.005  # 학습률을 너무 높게 설정하면 발산할 수 있고, 너무 낮으면 학습이 되지 않을 수 있습니다.

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden() 
    #입력과 목표 Tensor 생성

    rnn.zero_grad() 
    #0으로 초기화된 은닉 상태 생성

    #각 문자 읽기, 다음 문자를 위한 은닉 상태 유지
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    #목표와 최종 출력 비교
    loss.backward() 
    #역전파

    # 매개변수의 경사도에 학습률을 곱해서 그 매개변수의 값에 더합니다.
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()
    #출력, 손실 반환

#%%
import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000


# 도식화를 위한 손실 추적
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    #floor():실수를 입력하면 내림하여 정수를 반환
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

import wandb
wandb.login()
wandb.init(project='week6')

#%%
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # iter 숫자, 손실, 이름, 추측 화면 출력
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # 현재 평균 손실을 전체 손실 리스트에 추가
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

    if iter % 1000 == 0:
        wandb.log({'epoch': iter, 'loss':all_losses[int(iter/1000)-1]})
   
        

#%%
#결과 도식화
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

#%%
#결과 평가
# 혼란 행렬에서 정확한 추측을 추적
confusion = torch.zeros(n_categories, n_categories)
print(confusion)
n_confusion = 10000

# 주어진 라인의 출력 반환
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# 예시 중 어떤 것이 정확히 예측되었는지 기록
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# 모든 행을 합계로 나누어 정규화
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# 도식 설정
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 축 설정
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)
#set_xticklabels(), set_yticklabels() : X축과 Y축에 눈금이름을 설정

# 모든 tick에서 레이블 지정
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#set_major_locator:틱 간격 설정
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


#%%
#사용자 입력으로 실행
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        #topk 함수는 가장 큰 k개의 값과 인덱스 모두 반환
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            #item():딕셔너리의 내장함수를 통해서 키와 값을 포함하는 모든 맴버를 추출
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
