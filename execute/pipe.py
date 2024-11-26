import numpy as np
import torch
from utils import preprocess, infer
import threading

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../pipeline"))
from pipeline import pipe

def process_stream(device_path, buffer, seq_len, model, scaler):
    """
    실시간 데이터 스트림을 처리하고 모델 추론을 수행합니다.
    """
    
    stop_event = threading.Event()  # 중단 이벤트 생성

    for raw_data in pipe(device_path, stop_event):
        # 데이터 전처리
        preprocessed_data = preprocess(raw_data, scaler)
        if preprocessed_data is None:
            print("Invalid data. Skipping...")
            continue

        # 버퍼에 데이터 추가
        buffer.append(preprocessed_data)
        if len(buffer) > seq_len:
            buffer.pop(0)

        # 버퍼를 Tensor로 변환
        buffer_np = np.array(buffer, dtype=np.float32)
        input_seq = torch.tensor(buffer_np, dtype=torch.float32).unsqueeze(0).to("cuda")

        # 추론 수행
        result = infer(input_seq, model)

        # 결과 출력
        print(f"Reconstruction Error: {result['reconstruction_error']:.4f}")
        print(f"Real or Fake: {result['real_or_fake']}")
