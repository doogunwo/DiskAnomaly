import torch
import torch.nn as nn
import torch.optim as optim
from pipe_save import pipe
from model import AADModel
import numpy as np
import threading
import sys
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.path.join(os.path.dirname(__file__), "checkpoint"))
sys.path.append(os.path.join(os.path.dirname(__file__), "pipeline"))
# Parameters
input_dim = 4  # Number of features (Timestamp, IO_Type, Sector, Size)
hidden_dim = 64  # Hidden dimension of LSTM
latent_dim = 16  # Latent dimension of Encoder
seq_len = 10  # Sequence length for input data
threshold = 0.7  # Anomaly detection threshold

# Load model
model = AADModel(input_dim, hidden_dim, latent_dim).to(device)
model.load_state_dict(torch.load("./checkpoint/model.pth"))  # Load trained model checkpoint
model.eval()  # Set model to evaluation mode

# Preprocessing function
def preprocess(data):
    """
    Preprocess the raw data from the pipe function.
    Normalize and prepare for model input.
    """
    io_type_dict = {"A": 1, "Q": 2, "G": 3, "D": 4, "I": 5, "C": 6}
    io_type_numeric = io_type_dict.get(entry["IO_Type"], 0)
    normalized_data = []
    for entry in data:
        normalized_data.append([
            entry["Timestamp"],  # Example normalization for each feature
            io_type_numeric,
            entry["Sector"] / 1e9,  # Scale Sector value
            entry["Size"] / 4096  # Scale Size value
        ])
    return np.array(normalized_data, dtype=np.float32)

# Real-time data processing
def process_real_time_data(device):
    buffer = []  # Buffer to store incoming data
    stop_event = threading.Event()

    try:
        for raw_data in pipe(device, stop_event):
            buffer.append(raw_data)

            # Ensure buffer size matches sequence length
            if len(buffer) == seq_len:
                input_seq = preprocess(buffer)
                buffer.pop(0)  # Remove oldest data point to maintain sliding window

                # Convert to Tensor
                input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)

                with torch.no_grad():
                    # Model inference
                    _, reconstructed, z, _, _ = model(input_tensor, seq_len)

                    # Calculate reconstruction error as anomaly score
                    reconstruction_error = torch.mean((input_tensor - reconstructed) ** 2, dim=-1)
                    anomaly_score = torch.sigmoid(reconstruction_error)

                    # Print anomaly score
                    print(f"Anomaly Score: {anomaly_score.item()}")

                    # Check if anomaly score exceeds threshold
                    if anomaly_score > threshold:
                        print("Anomaly detected!")
                    else:
                        print("System is normal.")
    except KeyboardInterrupt:
        print("Real-time processing interrupted.")
        stop_event.set()

# Main execution
if __name__ == "__main__":
    device_to_monitor = "/dev/sda6"  # Change to your device
    process_real_time_data(device_to_monitor)

