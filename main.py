#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import signal
import threading
import torch
import numpy as np
import joblib
from utils import process_data_stream, data_queue, kill_blktrace
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
sys.path.append(os.path.join(os.path.dirname(__file__), "pipeline"))
from pipeline import pipe

sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model import AADModel

def generate_graphs(data_list):
    """
    종료 시 데이터를 기반으로 3개의 그래프를 생성합니다.
    """
    if not data_list:
        print("[INFO] No data to visualize.")
        return

    # 데이터 분리
    sequences = list(range(1, len(data_list) + 1))  # 1부터 시작하는 순서 번호
    anomalies = [d["Anomaly"] for d in data_list]
    io_types = [d["IO_Type"] for d in data_list]
    sizes = [d["Size"] for d in data_list]

    plt.figure(figsize=(18, 10))

    # 1. 이상치 선형 그래프
    plt.subplot(3, 1, 1)
    plt.plot(sequences, anomalies, label="Anomaly Score", color="red", alpha=0.7)
    plt.title("Anomaly Score Over Sequence")
    plt.xlabel("Sequence")
    plt.ylabel("Anomaly Score")
    plt.legend()

    # 2. 작업 빈도 그래프 (I/O Frequency)
    plt.subplot(3, 1, 2)
    io_counts = Counter(io_types)
    plt.bar(io_counts.keys(), io_counts.values(), color="blue", alpha=0.7)
    plt.title("I/O Frequency")
    plt.xlabel("I/O Type")
    plt.ylabel("Frequency")

    # 3. 요청 크기 히스토그램 (Request Size Distribution)
    plt.subplot(3, 1, 3)
    plt.hist(sizes, bins=20, color="green", alpha=0.7)
    plt.title("Request Size Distribution")
    plt.xlabel("Request Size")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("result.png")

def process_and_save_data(stop_event):
    """
    데이터 스트림을 처리하며 데이터를 data_list에 저장합니다.
    """
    print("[INFO] Starting data processing...")
    sequence_number = 0  # 순서를 기록할 카운터

    while not stop_event.is_set():
        while not data_queue.empty():
            data = data_queue.get()
            sequence_number += 1  # 순서 증가

            # 데이터에 순서 번호 추가
            data["Sequence"] = sequence_number
            data_list.append(data)

if __name__ == "__main__":
    # 모델 초기화 및 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load("./checkpoint/scaler.pkl")
    model_path = "./checkpoint/model_test.pth"
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model = AADModel(input_dim=4, hidden_dim=64, latent_dim=32).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    seq_len = 10  # 시퀀스 길이
    ssd = "/dev/sdb5"
    stop_event = threading.Event()  # 스레드 종료 플래그
    data_list = []

    stop_event = threading.Event()

    try:
        # 스레드 시작
        stream_thread = threading.Thread(target=process_data_stream, args=(ssd, seq_len, model, scaler, stop_event))
        process_thread = threading.Thread(target=process_and_save_data, args=(stop_event,))

        stream_thread.start()
        process_thread.start()

        print("[INFO] Running... Press Ctrl+C to stop and generate graphs.")
        # 메인 스레드가 대기 상태에 있음
        stream_thread.join()
        process_thread.join()

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected. Stopping threads...")
        stop_event.set()  # 모든 스레드 종료 신호
        stream_thread.join()
        process_thread.join()
        kill_blktrace()

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        stop_event.set()
        stream_thread.join()
        process_thread.join()

    finally:
        generate_graphs(data_list)
        print("[INFO] Exiting program.")

