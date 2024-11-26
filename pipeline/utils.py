import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))
from model import AADModel  # 모델 정의
from sklearn.preprocessing import MinMaxScaler

class ModelHandler:
    def __init__(self, model_path, device, input_dim=4, hidden_dim=64, latent_dim=32):
        self.device = device
        self.model = AADModel(input_dim, hidden_dim, latent_dim).to(self.device)
        self.scaler = MinMaxScaler()
        self.load_model(model_path)

    def load_model(self, model_path):
       
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set model to evaluation mode

    def preprocess(self, data):
       
        io_type_dict = {"A": 1, "Q": 2, "G": 3, "D": 4, "I": 5, "C": 6}
        preprocessed_data = []
        for entry in data:
            io_type_numeric = io_type_dict.get(entry["IO_Type"], 0)
            preprocessed_data.append([
                entry["Timestamp"],
                io_type_numeric,
                entry["Sector"] / 1e9,
                entry["Size"] / 4096
            ])
        normalized = self.scaler.fit_transform(preprocessed_data)
        return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

    def compute_anomaly_score(self, input_seq):
        
        with torch.no_grad():
            reconstructed, _, _, _, _ = self.model(input_seq, seq_len=10)
            reconstruction_error = torch.mean((input_seq - reconstructed) ** 2, dim=-1).item()
            return torch.sigmoid(torch.tensor(reconstruction_error)).item()
