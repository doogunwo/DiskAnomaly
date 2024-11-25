from sklearn.preprocessing import MinMaxScaler
import sys
import os
import threading
import numpy as np
import torch
import preprocessing
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from preprocessing import preprocess_dataset  # 함수 임포트
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), "pipeline"))
from pipeline import pipe
from utils import ModelHandler
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model import AADModel
import signal
import os
import signal
import psutil
import atexit

# Global settings
seq_len = 10  # Sequence length expected by the model
buffer = []  # 모델 입력을 위한 슬라이딩 윈도우

def find_and_kill_blktrace():
    """
    실행 중인 blktrace 프로세스를 찾아 종료합니다.
    """
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if 'blktrace' in proc.info['name']:
                print(f"Terminating blktrace process with PID: {proc.info['pid']}")
                os.kill(proc.info['pid'], signal.SIGINT)  # SIGINT로 종료 시도
    except Exception as e:
        print(f"Error while terminating blktrace: {e}")

def preprocess(data, scaler):
    """
    실시간 데이터를 전처리합니다.
    Args:
        data (dict): 실시간으로 수집된 데이터 포인트.
        scaler (MinMaxScaler): 훈련 시 사용된 스케일러.
    Returns:
        np.array: 전처리된 데이터 포인트.
    """
    io_type_map = {"A": 0, "Q": 1, "G": 2, "I": 3, "D": 4, "C": 5}
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

        
if __name__ == "__main__":
    # Initialize the model handler
    device = "cuda" 
    model_handler = ModelHandler("./checkpoint/model_test.pth", device)
    scaler_path = "./checkpoint/scaler.pkl"

    model_handler.model = AADModel(input_dim=4, hidden_dim=4, latent_dim=4).to(device)
    # Start real-time anomaly detection
    print("학습 데이터로부터 스칼라 저장")
    stop_event = threading.Event()
    scaler_load = joblib.load(scaler_path)
    
    print("디스크 서칭 시작")
    try:
        for raw_data in pipe("/dev/sda6", stop_event):
            try:
                # 실시간 데이터 전처리
                preprocessed_data = preprocess(raw_data, scaler_load)

                if preprocessed_data is None:
                    print("Preprocessed data is None. Skipping...")
                    continue

                buffer.append(preprocessed_data)

                if len(buffer) > seq_len:
                    buffer.pop(0)

                if len(buffer) == seq_len:
                    buffer_np = np.array(buffer)
                    input_seq = torch.tensor(buffer_np, dtype=torch.float32).unsqueeze(0).to(device)

                    print(f"Input sequence shape: {input_seq.shape}")  # 디버깅용

                    seq_len_actual = input_seq.size(1)

                    # 디코더 호출
                    try:
                        reconstructed = model_handler.model.decoder(input_seq, seq_len_actual)
                        print(f"Reconstructed shape: {reconstructed.shape}")

                        reconstruction_error = torch.mean((input_seq - reconstructed) ** 2).item()
                        print(f"Reconstruction Error: {reconstruction_error:.4f}")
                    except Exception as e:
                        print(f"Error occurred: {type(e).__name__}: {e}")
                    
            except Exception as e:
                # 예외 종류와 상세 메시지 출력
                print(f"Error occurred: {type(e).__name__}: {e}")
                continue

    except KeyboardInterrupt:
        print("\nStopping anomaly detection...")
        stop_event.set()

    # Register process cleanup
    atexit.register(find_and_kill_blktrace)
