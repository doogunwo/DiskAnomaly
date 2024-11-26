import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm
from joblib import Parallel, delayed


def preprocess_dataset(file_path: str, seq_len: int):
    # Step 1: Load data
    print("Loading dataset...")
    df = pd.read_csv(
        file_path,
        header=0,  # 첫 번째 줄을 헤더로 사용
        names=["Timestamp", "IO_Type", "Sector", "Size"],
        dtype={"Timestamp": float, "IO_Type": str, "Sector": int, "Size": int},
        low_memory=False
    )

    # Step 2: Encode IO_Type (Categorical to Numerical)
    print("Encoding IO_Type...")
    df["IO_Type"] = df["IO_Type"].map({"A": 0, "Q": 1, "G": 2, "I": 3, "D": 4, "C": 5, "M": 6})
        
    
    df = df.dropna()  # Drop rows with NaN values

    # Step 3: Scale the numeric columns
    print("Scaling numeric data...")
    scaler = MinMaxScaler()
    df[["Timestamp", "Sector", "Size"]] = scaler.fit_transform(df[["Timestamp", "Sector", "Size"]])

    # Step 4: Prepare sequences for LSTM using parallel processing
    print("Preparing sequences...")
    batch_size = 10000
    array = df.values

    # sampleing
    sequences = []
    for start in tqdm(range(0, len(array) - seq_len + 1, batch_size), desc="Preparing Sequences", unit="batch"):
        end = min(start + batch_size, len(array) - seq_len + 1)
        batch_sequences = [array[i:i+seq_len] for i in range(start, end)]
        sequences.extend(batch_sequences)

    # Convert to PyTorch Tensor
    print("Converting to PyTorch Tensor...")
    data_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)

    return data_tensor, scaler, df

def preprocess_dataset_sampleing(file_path: str, seq_len: int):
    # Step 1: Load data
    print("Loading dataset...")
    df = pd.read_csv(
        file_path,
        header=0,  # 첫 번째 줄을 헤더로 사용
        names=["Timestamp", "IO_Type", "Sector", "Size"],
        dtype={"Timestamp": float, "IO_Type": str, "Sector": int, "Size": int},
        low_memory=False
    )

    # Step 2: Encode IO_Type (Categorical to Numerical)
    print("Encoding IO_Type...")
    df["IO_Type"] = df["IO_Type"].map({"A": 0, "Q": 1, "G": 2, "I": 3, "D": 4, "C": 5, "M": 6})
    
    df = df.dropna()  # Drop rows with NaN values

    # Step 3: Scale the numeric columns
    print("Scaling numeric data...")
    scaler = MinMaxScaler()
    df[["Timestamp", "Sector", "Size"]] = scaler.fit_transform(df[["Timestamp", "Sector", "Size"]])

    # Step 4: Prepare sequences for LSTM using parallel processing
    print("Preparing sequences...")
    batch_size = 518
    array = df.values
    sequences = []
    for start in tqdm(range(0, len(array) - seq_len + 1, batch_size), desc="Preparing Sequences", unit="batch"):
            sequence = array[start:start +seq_len]
            sequences.append(sequence)

# Convert to PyTorch Tensor
    print("Converting to PyTorch Tensor...")
    data_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)

    return data_tensor, scaler, df
