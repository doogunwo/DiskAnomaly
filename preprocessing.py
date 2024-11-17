import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm

def preprocess_dataset(file_path: str, seq_len: int):
   
    # Step 1: Load data
    print("Loading dataset...")
    with tqdm(total=1, desc="Loading Dataset") as pbar:
        df = pd.read_csv(file_path, names=["Timestamp", "IO_Type", "Sector", "Size"])
        pbar.update(1)

    # Step 2: Encode IO_Type (Categorical to Numerical)
    print("Encoding IO_Type...")
    with tqdm(total=1, desc="Encoding IO_Type") as pbar:
        df["IO_Type"] = df["IO_Type"].map({"A": 0, "Q": 1, "G": 2, "I": 3, "D": 4, "C": 5})
        df = df.dropna()  # Drop rows with NaN values
        pbar.update(1)

    # Step 3: Scale the numeric columns
    print("Scaling numeric data...")
    with tqdm(total=1, desc="Scaling Data") as pbar:
        scaler = MinMaxScaler()
        df[["Timestamp", "Sector", "Size"]] = scaler.fit_transform(df[["Timestamp", "Sector", "Size"]])
        pbar.update(1)

    # Step 4: Prepare sequences for LSTM
    print("Preparing sequences...")
    sequences = []
    with tqdm(total=len(df) - seq_len + 1, desc="Preparing Sequences") as pbar:
        for i in range(len(df) - seq_len + 1):
            sequence = df.iloc[i:i+seq_len].values
            sequences.append(sequence)
            pbar.update(1)

    # Convert to PyTorch Tensor
    print("Converting to PyTorch Tensor...")
    with tqdm(total=1, desc="Converting to Tensor") as pbar:
        data_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
        pbar.update(1)

    return data_tensor, scaler, df
