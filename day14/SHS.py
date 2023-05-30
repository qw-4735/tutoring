

#%%
#토큰화(Tokenization)
from transformers import BertTokenizer
#tok = BertTokenizer.from_pretrained("bert-base-uncased")
'''
토큰화 과정에서 NotImplementedError: Cannot 에러 발생
버전 문제?
'''

tok = BertTokenizer
tok("Hello, how are you doing?")['inputs_ids']
tok("The Frenchman spoke in the [MASK] language and ate 🥖")['input_ids']
tok("[CLS] [SEP] [MASK] [UNK]")['input_ids']

#토크나이저는 자동으로 인코딩의 시작 ([CLS]=101[CLS] = 101[CLS]=101)및 인코딩의 종료(즉, 분리) ([SEP]=102[SEP] = 102[SEP]=102)에 대한 토큰을 포함
#마스킹 ([MASK]=103[MASK] = 103[MASK]=103) 및 알 수 없는 기호 ([UNK]=100) 




#%%
#임베딩(Embeddings)
#임베딩을 통해 벡터로 변환->정규화
import math
import torch
from torch import nn
class Embed(nn.Module):
    def __init__(self, vocab: int, d_model: int = 512):
        super(Embed, self).__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.emb = nn.Embedding(self.vocab, self.d_model)
        #어휘의 크기 및 모델의 차원 지정
        #torch.nn.Embedding(num_embeddings, embedding_dim)
        #Input: IntTensor or LongTensor 
        
        self.scaling = math.sqrt(self.d_model) #정규화

    def forward(self, x):
        return self.emb(x) * self.scaling


#%%    
#포지셔널 인코딩(Positional Encoding)
'''
Transformer는 RNN을 사용하지 않기 때문에, 문장의 위치 정보를 따로 제공해줄 필요
인코더와 디코더에 대한 입력 임베딩에 인코딩을 추가->임베드된 토큰의 상대 위치 입력
포지셔닝 인코딩은 숫자 오버 플로우 (numerical overflow) 방지를 위해 로그 공간(log space)에서 연산
'''
import torch
from torch import nn
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = .1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Compute the positional encodings in log space
        #비어있는 tensor 생성
        pe = torch.zeros(max_len, d_model)
        #position(0-max_len) 생성 (row 바양 -> unsqueeze(dim=1))
        position = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.Tensor([10000.0])) / d_model))
        
        # (i가 짝수일때 : sin, 홀수일때 : cos)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [max_len, hid_dim] -> [1, max_len, hid_dim]
        pe = pe.unsqueeze(0)
        # 매개 변수로 간주되지 않는 버퍼를 등록하는 데 사용
        #optimizer가 update하지 않는다. / state_dict에는 저장된다. / GPU에서 작동한다.
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        #pe[:, :out1.shape[1]] : x의 shape에 맞출 수 있도록 indexing
        return self.dropout(x)



#%%
#다중 헤드 어텐션(Multi-Head Attention)
'''
Attention은 "주어진 쿼리(Query)에 대해서 모든 키(Key)와의 유사도를 각각 구해 키와 매핑되어 있는 값(Value)에 반영" 
어텐션 레이어(attention layer)는 쿼리(Q)와 키(K), 값(V)쌍 간의 맵핑을 학습
쿼리(query)는 입력의 임베딩이며, 값(value)과 키(key)는 타깃 ex)유튜브검색창:query, 동영상:value
'''
import torch
from torch import nn
class Attention:
    def __init__(self, dropout: float = 0.):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    # Q, K, V를 모두 입력문장으로부터 가져온다
    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        
        #scores : scale dot product attention
        #query, key를 행렬곱
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # Mask가 있다면 마스킹된 부위 -1e9으로 채우기
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            #masked_fill : 마스크에서 특정값을 바꿔준다
        p_attn = self.dropout(self.softmax(scores))
        return torch.matmul(p_attn, value)
    
    def __call__(self, query, key, value, mask=None):
        #nn.Module을 상속받은 class에서 forward와 동일한 효과를 낸다
        return self.forward(query, key, value, mask)
'''
#단일 어텐션 레이어는 하나의 표현만을 허용->트랜스포머에서 다중 어텐션 헤드(multiple attention heads)가 사용

논문에서는 결합된(concatenated) h=8h = 8h=8 어텐션 레이어를 사용
MultiHead(Q,K,V)=Concat(head1,...,headn)WO
where headi=Attention(QWiQ,KWiK,VWiV)head i = Attention(QWi_Q, KWi_K, VWi_V)
'''


from torch import nn
from copy import deepcopy
class MultiHeadAttention(nn.Module):
    def __init__(self, h: int = 8, d_model: int = 512, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.attn = Attention(dropout)
        self.lindim = (d_model, d_model)
        self.linears = nn.ModuleList([deepcopy(nn.Linear(*self.lindim)) for _ in range(4)])
        self.final_linear = nn.Linear(*self.lindim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        #Q,K,V를  d_k, d_k, d_v 차원으로 projection -> Q,K,V를 head 수 만큼 분리
        #   -> (batch_size, head, seq_len, head_dim)
        query, key, value = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2) \
                             for l, x in zip(self.linears, (query, key, value))]
        
        '''
        q = nn.Linear(hid_dim, hid_dim)(q)
        k = nn.Linear(hid_dim, hid_dim)(k)
        v = nn.Linear(hid_dim, hid_dim)(v)
        # 여기까지는 위 코드처럼 바꾸었을 때 다음과 같다.
        # q, k, v = [l(x) for l, x in zip(linears, (q, k, v))]
        
        q = q.view(query.size(0), -1, n_heads, head_dim)
        k = q.view(query.size(0), -1, n_heads, head_dim)
        v = q.view(query.size(0), -1, n_heads, head_dim)
        # 여기까지를 위 코드처럼 바꾸었을 때 다음과 같다.
        # q, k, v = [l(x).view(query.size(0), -1, n_heads, head_dim) for l, x in zip(linears, (q, k, v))]
        
        q = q.transpose(1, 2)
        k = q.transpose(1, 2)
        v = q.transpose(1, 2)
        # 여기까지 위 코드처럼 바꾸었을 때 다음과 같다.
        # q, k, v = [l(x).view(bs, -1, n_heads, head_dim).transpose(1, 2) for l, x in zip(linears, (q, k, v))]
        '''
        nbatches = query.size(0)
        
        #Scaled Dot-Product Attention 을 수행
        x = self.attn(query, key, value, mask=mask)
        
 
        #분리된 head들을 concat 하기 -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        #.transpose는 기본 메모리 저장소를 원래의 tensor와 공유하기 때문에 .contiguous 방법은 .transpose 다음에 추가
        #.view를 호출하려면 인접한(contiguous) tensor가 필요
        return self.final_linear(x)

#%%
#레지듀얼 및 레이어 정규화(Residuals and Layer Normalization)
#레이어 정규화
from torch import nn
class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
#레지듀얼
#네트워크에서 이전 레이어의 출력을 (즉, 하위 레이어) 현재 레이어의 출력에 추가 ->very deep network
#residualconnection(x)=x+Dropout(SubLayer(LayerNorm(x)))
from torch import nn
class ResidualConnection(nn.Module):
    def __init__(self, size: int = 512, dropout: float = .1):
        super(ResidualConnection,  self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) 

#%%
#피드 포워드(Feed Forward)
# ReLUReLUReLU 활성화(activation) (ReLU(x)=max(0,x)ReLU(x) = max(0, x)ReLU(x)=max(0,x))) 와 내부 레이어 드롭아웃을 통해 완전히 연결된 (fully-connected) 레이어로 구성
#FeedForward(x) = W2max(0,xW1+B1)+B2
from torch import nn
class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = .1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.dropout(self.relu(self.l1(x))))
 
#%%   
#인코더-디코더
#인코더
#Encoding(x,mask)=FeedForward(MultiHeadAttention(x))
from torch import nn
from copy import deepcopy
class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float = .1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sub1 = ResidualConnection(size, dropout)
        self.sub2 = ResidualConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        # Multi-Head Attention -> residual connection
        x = self.sub1(x, lambda x: self.self_attn(x, x, x, mask))
        # Multi Head Attention 에서 Query, Key, Value 자리에 동일한 x ->문장 내의 토큰끼리 관계도를 연산 목적 
        return self.sub2(x, self.feed_forward)
    '''
    # Input x: [batch_size, seq_len, hid_dim] 
    # Output: 동일한 Shape
    '''
#인코더는 레이어 정규화가 뒤따르는 6개의 동일한 레이어 정규화로 구성
class Encoder(nn.Module):
    def __init__(self, layer, n: int = 6):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n)])
        '''
        x가 layer 입력으로 들어가고 그에 대한 결과물을 x로 받는다.
        for 문 안에서 이 x가 다시 layer의 입력으로 들어가게 된다.
        이런 방법으로 EncoderLayer 6개 생성
        '''
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    # output x: [batch_size, seq_len, hid_dim]
#%%
#디코더
'''
메모리를 포함한 다중 헤드 어텐션 레이어가 뒤따르는 마스킹된 (maksed) 다중 헤드 어텐션 레이어
-> 메모리는 인코더의 출력
'''
#Decoding(x, memory, mask1, mask2) = FeedForward(MultiHeadAttention(MultiHeadAttention(x,mask1),,memory,mask2))
from torch import nn
from copy import deepcopy
class DecoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadAttention, src_attn: MultiHeadAttention, 
                 feed_forward: FeedForward, dropout: float = .1):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sub1 = ResidualConnection(size, dropout)
        self.sub2 = ResidualConnection(size, dropout)
        self.sub3 = ResidualConnection(size, dropout)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        #1. masked multi-head attention (self attention) + add
        x = self.sub1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        
        #2. multi-head attention (encoder-decoder attention)
        x = self.sub2(x, lambda x: self.src_attn(x, memory, memory, src_mask))
        #query는 decoder에서 올라오고, key, value는 encoder에서 넘어온 값(memory)을 사용
        
        return self.sub3(x, self.feed_forward)

#디코더는 레이어 정규화가 뒤따르는 6개의 동일한 레이어를 가지고 있다
class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, n: int = 6):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Output(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Output, self).__init__()
        self.l1 = nn.Linear(input_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.l1(x)
        return self.log_softmax(logits)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: Embed, tgt_embed: Embed, final_layer: Output):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.final_layer = final_layer
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.final_layer(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

#%%
#최종 출력
#디코더의 벡터 출력을 최종 출력으로 변환
#디코더 출력->logits의 행렬로, 이는 타깃 어휘의 차원을 갖고 있음
#->softmax 활성화 함수를 통해 어휘에 대한 확률 분포로 변환
#Output(x) = LogSoftmax(max(0,xW1+B1))

#%%    
#모델 초기화
from torch import nn
def make_model(input_vocab: int, output_vocab: int, d_model: int = 512):
    encoder = Encoder(EncoderLayer(d_model, MultiHeadAttention(), FeedForward()))
    decoder = Decoder(DecoderLayer(d_model, MultiHeadAttention(), MultiHeadAttention(), FeedForward()))
    input_embed= nn.Sequential(Embed(vocab=input_vocab), PositionalEncoding())
    output_embed = nn.Sequential(Embed(vocab=output_vocab), PositionalEncoding())
    output = Output(input_dim=d_model, output_dim=output_vocab)
    model = EncoderDecoder(encoder, decoder, input_embed, output_embed, output)
    
    # Initialize parameters with Xavier uniform 
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

#%%
# Tokenized symbols for source and target.
src = torch.tensor([[1, 2, 3, 4, 5]])
src_mask = torch.tensor([[1, 1, 1, 1, 1]])
tgt = torch.tensor([[6, 7, 8, 0, 0]])
tgt_mask = torch.tensor([[1, 1, 1, 0, 0]])

# Create PyTorch model
model = make_model(input_vocab=10, output_vocab=10)
#입력과 출력에 대한 어휘가 10개만 있다고 가정

# Do inference and take tokens with highest probability through argmax along the vocabulary axis (-1)
result = model(src, tgt, src_mask, tgt_mask)
result.argmax(dim=-1) #tensor([[4, 4, 3, 4, 4]])
