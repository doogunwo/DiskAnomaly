import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # Import tqdm for progress visualization

from preprocessing import preprocess_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model import AADModel

if __name__ == "__main__":
    # Hyperparameters
    input_dim = 4  # Timestamp, IO_Type, Sector, Size
    hidden_dim = 64
    latent_dim = 32
    seq_len = 10
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    print("Data preprocessing")
    # Preprocess dataset
    file_path = "./data/blktrace_data.csv"
    with tqdm(total=1, desc="Preprocessing Dataset") as pbar:
        data, _, _ = preprocess_dataset(file_path, seq_len)
        pbar.update(1)

    # Create DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model
    model = AADModel(input_dim, hidden_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Example loss function
    print("Model training start")

    # Training Loop
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_bar:
            for batch in batch_bar:
                inputs = batch[0]
                seq_len = inputs.size(1)

                # Forward pass
                reconstructed, real_or_fake, z, mu, logvar = model(inputs, seq_len)

                # Compute loss (reconstruction + KL divergence example)
                recon_loss = criterion(reconstructed, inputs)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_bar.set_postfix(Loss=loss.item())  # Update batch progress bar

        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}")
