import sys
import os
import threading
import numpy as np
import torch
import joblib
import pandas as pd
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "pipeline"))
from pipeline import pipe

sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model import AADModel

# 설정
seq_len = 10  # 시퀀스 길이
buffer = []  # 슬라이딩 윈도우용 버퍼
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model_path = "./checkpoint/model_test.pth"
checkpoint = torch.load(model_path, map_location=device,weights_only=True)

model = AADModel(input_dim=4, hidden_dim=64, latent_dim=32).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 스케일러 로드
scaler = joblib.load("./checkpoint/scaler.pkl")  # 학습 시 사용한 스케일러

def preprocess(data, scaler):
    """
    실시간 데이터를 전처리합니다.
    Args:
        data (dict): 실시간으로 수집된 데이터 포인트.
        scaler (MinMaxScaler): 훈련 시 사용된 스케일러.
    Returns:
        np.array: 전처리된 데이터 포인트.
    """
    print("전처리")
    io_type_map = {"A": 0, "Q": 1, "G": 2, "I": 3, "D": 4, "C": 5 , "M": 6}
    io_type = io_type_map.get(data["IO_Type"], -1)
    if io_type == -1:
        raise ValueError(f"Unknown IO_Type: {data['IO_Type']}")

    # `Timestamp`, `Sector`, `Size`만 스케일링
    input_features = pd.DataFrame([[data["Timestamp"], data["Sector"], data["Size"]]], 
                                   columns=["Timestamp", "Sector", "Size"])  # Feature names 추가
    try:
        scaled_values = scaler.transform(input_features)[0]
    except Exception as e:
        print(f"Scaling error: {e}")
        return None

    # 스케일링된 값 + IO_Type 결합
    preprocessed_data = np.array([scaled_values[0], io_type, scaled_values[1], scaled_values[2]])
    return preprocessed_data

# 추론 함수
def infer(input_seq, model):
    seq_len_actual = input_seq.size(1)
    with torch.no_grad():
        z, mu, logvar = model.encoder(input_seq)
        reconstructed = model.decoder(z, seq_len_actual)
        reconstruction_error = torch.mean((input_seq - reconstructed) ** 2).item()
        real_or_fake = model.discriminator(input_seq)
        return {
            "reconstructed": reconstructed.cpu().numpy(),
            "reconstruction_error": reconstruction_error,
            "real_or_fake": real_or_fake.cpu().numpy()
        }

# 데이터 스트림 처리
def process_stream(device_path, buffer, seq_len, model, scaler):
    """
    실시간 데이터 스트림을 처리하고 모델 추론을 수행합니다.
    """
    stop_event = threading.Event()  # 중단 이벤트 생성
    for raw_data in pipe(device_path, stop_event):
        # 데이터 전처리
        print("raw_data in pipe")
        preprocessed_data = preprocess(raw_data, scaler)
        print("preprocess")
        if preprocessed_data is None:
            print("Invalid data. Skipping...")
            continue

        # 버퍼에 데이터 추가
        print("add buffer")
        buffer.append(preprocessed_data)

        if len(buffer) > seq_len:
            buffer.pop(0)

        buffer_np = np.array(buffer, dtype=np.float32)
        input_seq = torch.tensor(buffer_np, dtype=torch.float32).unsqueeze(0).to(device)
        
            # 추론 수행
        result = infer(input_seq, model)

            # 결과 출력
        print(f"Reconstruction Error: {result['reconstruction_error']:.4f}")
        print(f"Real or Fake: {result['real_or_fake']}")
       
# `pipe` 함수와 통합 실행
if __name__ == "__main__":
    try:
        device_path = "/dev/sda6"
        process_stream(device_path, buffer, seq_len, model, scaler)
    except KeyboardInterrupt:
        print("Real-time processing stopped by user.")
