import torch


# 설정
seq_len = 10  # 시퀀스 길이
buffer = []  # 슬라이딩 윈도우용 버퍼
device_path = "/dev/sda6"  # 디바이스 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 디바이스 설정
model_path = "../checkpoint/model_test.pth"
scaler_path = "../checkpoint/scaler.pkl"