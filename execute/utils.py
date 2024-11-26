import torch
import joblib
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))
from model import AADModel

def load_model(model_path, scaler_path):
    """
    모델과 스케일러를 로드합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = AADModel(input_dim=4, hidden_dim=64, latent_dim=32).to(device)
    checkpoint = torch.load(model_path, map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 스케일러 로드
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess(data, scaler):
    """
    실시간 데이터를 전처리합니다.
    """
    io_type_map = {"A": 0, "Q": 1, "G": 2, "I": 3, "D": 4, "C": 5, "M": 6}
    io_type = io_type_map.get(data["IO_Type"], -1)
    if io_type == -1:
        raise ValueError(f"Unknown IO_Type: {data['IO_Type']}")

    # 스케일링
    input_features = pd.DataFrame([[data["Timestamp"], data["Sector"], data["Size"]]], 
                                   columns=["Timestamp", "Sector", "Size"])
    try:
        scaled_values = scaler.transform(input_features)[0]
    except Exception as e:
        print(f"Scaling error: {e}")
        return None

    preprocessed_data = np.array([scaled_values[0], io_type, scaled_values[1], scaled_values[2]])
    return preprocessed_data

def infer(input_seq, model):
    """
    모델 추론을 수행합니다.
    """
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
