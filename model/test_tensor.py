import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from torchtext.vocab import GloVe

# 예제 텍스트 데이터
texts = ["This is an example.", "Another example text!", "Text preprocessing for LSTM."]

# Step 1: 토큰화
tokenized_texts = [text.lower().split() for text in texts]  # 단순 단어 분리

# Step 2: 텍스트를 인덱스로 변환
vocab = set(word for tokens in tokenized_texts for word in tokens)  # 고유 단어집 생성
word_to_index = {word: idx for idx, word in enumerate(vocab)}  # 단어 -> 인덱스 매핑

# 텍스트 데이터를 인덱스 시퀀스로 변환
indexed_texts = [[word_to_index[word] for word in tokens] for tokens in tokenized_texts]

# Step 3: 패딩 및 시퀀스 길이 통일
sequence_length = 10  # 원하는 시퀀스 길이
padded_sequences = [seq[:sequence_length] + [0] * (sequence_length - len(seq)) for seq in indexed_texts]

# Step 4: 임베딩 적용 (GloVe를 사용한 예시)
embedding_dim = 50  # GloVe 임베딩 차원 (예: 50차원)
glove = GloVe(name='6B', dim=embedding_dim)  # 사전 훈련된 GloVe 임베딩 로드

# 임베딩 벡터로 변환
embedded_sequences = [
    torch.stack([glove[word] if word in glove.stoi else torch.zeros(embedding_dim) for word in tokens])
    for tokens in padded_sequences
]

# Step 5: 텐서 형성 (batch_size, sequence_length, embedding_dim)
input_tensor = torch.stack(embedded_sequences)

print("입력 텐서 형태:", input_tensor.shape)  # (batch_size, sequence_length, embedding_dim)
