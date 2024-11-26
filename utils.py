import sys
import os
import threading
import numpy as np
import torch
from queue import Queue
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import joblib
import pandas as pd
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "pipeline"))
from pipeline import pipe

sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model import AADModel

# Queue 생성
data_queue = Queue(maxsize=50000)

def preprocess(data, scaler):
    """
    실시간 데이터를 전처리합니다.
    """
    io_type_map = {"A": 0, "Q": 1, "G": 2, "I": 3, "D": 4, "C": 5, "M": 6}
    io_type = io_type_map.get(data["IO_Type"], -1)
    if io_type == -1:
        raise ValueError(f"Unknown IO_Type: {data['IO_Type']}")

    # `Timestamp`, `Sector`, `Size`만 스케일링
    input_features = pd.DataFrame([[data["Timestamp"], data["Sector"], data["Size"]]], 
                                   columns=["Timestamp", "Sector", "Size"])
    scaled_values = scaler.transform(input_features)[0]
    preprocessed_data = np.array([scaled_values[0], io_type, scaled_values[1], scaled_values[2]])
    return preprocessed_data

def infer(input_seq, model):
    """
    모델 추론 수행.
    """
    seq_len_actual = input_seq.size(1)
    with torch.no_grad():
        z, mu, logvar = model.encoder(input_seq)
        reconstructed = model.decoder(z, seq_len_actual)
        reconstruction_error = torch.mean((input_seq - reconstructed) ** 2).item()
        real_or_fake = model.discriminator(input_seq)
        return reconstruction_error, real_or_fake.cpu().numpy()

def process_data_stream(device_path, seq_len, model, scaler):
    """
    실시간 데이터 스트림을 처리하고 Flask 대시보드에 데이터를 업데이트합니다.
    """
    stop_event = threading.Event()
    buffer = []

    for raw_data in pipe(device_path, stop_event):
        try:
            # 데이터 전처리
            preprocessed_data = preprocess(raw_data, scaler)
            buffer.append(preprocessed_data)
            if len(buffer) > seq_len:
                buffer.pop(0)

            if len(buffer) == seq_len:
                buffer_np = np.array(buffer, dtype=np.float32)
                input_seq = torch.tensor(buffer_np, dtype=torch.float32).unsqueeze(0).to(device)
                _, anomaly = infer(input_seq, model)
                
                # Read I/O와 Write I/O 계산 (MB 단위)
                read_io = raw_data["Size"] / (1024 * 1024) if raw_data["IO_Type"] == "R" else 0
                write_io = raw_data["Size"] / (1024 * 1024) if raw_data["IO_Type"] == "W" else 0

                latest_data = {
                    "Timestamp": raw_data["Timestamp"],
                    "Read_IO": read_io,
                    "Write_IO": write_io,
                    "Anomaly": float(anomaly)
                }

                if not data_queue.full():
                    data_queue.put(latest_data)

        except Exception as e:
            print(f"Error in processing data: {e}")

import os
import signal
import psutil


def kill_blktrace():
    """
    Find and terminate all blktrace and blkparse processes.
    """
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] in ['blktrace', 'blkparse']:
            print(f"Terminating {proc.info['name']} process with PID: {proc.info['pid']}")
            os.kill(proc.info['pid'], signal.SIGTERM)
