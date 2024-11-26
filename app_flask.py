from flask import Flask, render_template, jsonify
import signal
import atexit

import torch, sys, os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
sys.path.append(os.path.join(os.path.dirname(__file__), "templates"))
from model import AADModel
import threading
from utils import process_data_stream, kill_blktrace, data_queue
import numpy as np
model = AADModel(input_dim=4, hidden_dim=64, latent_dim=32).to(device)

buffer = []
seq_len = 10  # 시퀀스 길이
data_buffer = {
    "timestamps": [],
    "io_types": [],
    "sectors": [],
    "sizes": [],
    "anomalies": [],
}

model_path = "./checkpoint/model_test.pth"
scaler = joblib.load("./checkpoint/scaler.pkl")  # 학습 시 사용한 스케일러
checkpoint = torch.load(model_path, map_location=device, weights_only=True)

app = Flask(__name__)

ssd = "/dev/sda6"
stream_thread = threading.Thread(target=process_data_stream, args=(ssd, seq_len, model, scaler))
stream_thread.start()

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    """
    Queue에 저장된 데이터를 JSON으로 반환.
    """
    data = []
    while not data_queue.empty():
        item = data_queue.get()
        # NumPy 배열 변환
        anomaly = item.get("Anomaly", 0)
        if isinstance(anomaly, np.ndarray):
            anomaly = anomaly.item()  # 배열에서 단일 값을 명확히 추출

        data.append({
            "Timestamp": item.get("Timestamp", "N/A"),
            "Read_IO": item.get("Read_IO", 0),
            "Write_IO": item.get("Write_IO", 0),
            "Anomaly": float(anomaly)  # 단일 값으로 변환
        })
        print("Sending data:", data)
    return jsonify(data)


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True,threaded=True,use_reloader=False)
    except KeyboardInterrupt:
        atexit.register(kill_blktrace)
        signal.signal(signal.SIGINT, lambda signum, frame: exit(0))
        signal.signal(signal.SIGTERM, lambda signum, frame: exit(0))    
        stream_thread.join()