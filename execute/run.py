import threading
import torch
from pipe import process_stream
from utils import load_model
from config import device_path, seq_len, buffer, model_path, scaler_path

def run():
    """
    Main function to initialize the model, load data, and process the stream.
    """
    # 모델과 스케일러 로드
    model, scaler = load_model(model_path, scaler_path)

    try:
        # 실시간 데이터 스트림 처리 시작
        process_stream(device_path, buffer, seq_len, model, scaler)
    except KeyboardInterrupt:
        print("Real-time processing stopped by user.")

if __name__ == "__main__":
    run()
