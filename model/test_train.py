import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocessing import preprocess_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model import AADModel

def find_optimal_lr(model, dataloader, criterion, device):
    """
    Finds the optimal learning rate using a range test.
    """
    learning_rates = torch.logspace(-5, 0, steps=100)  # Learning rate range: 1e-5 ~ 1
    losses = []

    # Test different learning rates
    for lr in learning_rates:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch[0].to(device)
            outputs, _, _, _, _ = model(inputs, seq_len)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            break  # Test one batch per learning rate

    # Plot loss vs learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates.numpy(), losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.show()

    # Find the optimal learning rate (e.g., loss starts decreasing)
    min_loss_idx = losses.index(min(losses))
    optimal_lr = learning_rates[min_loss_idx].item()
    return optimal_lr

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 파라미터
    input_dim = 4  # Timestamp, IO_Type, Sector, Size
    hidden_dim = 64
    latent_dim = 32
    seq_len = 10
    batch_size = 128
    num_epochs = 1

    print("Data preprocessing")
    # Preprocess dataset
    # 데이터 읽어 들이기.
    file_path = "./data/blktrace_data.csv"
    with tqdm(total=1, desc="Preprocessing Dataset") as pbar:
        data, _, _ = preprocess_dataset(file_path, seq_len)
        pbar.update(1)
    # 데이터로드 생성
    # Create DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 선언
    model = AADModel(input_dim, hidden_dim, latent_dim).to(device)
    criterion = nn.MSELoss()  # 로스함수

    # 최적의 학습률찾기
    print("Finding optimal learning rate...")
    optimal_lr = find_optimal_lr(model, dataloader, criterion, device)
    print(f"Optimal Learning Rate: {optimal_lr}")

    # 옵티마이저
    optimizer = torch.optim.RMSprop(model.parameters(), lr=optimal_lr)

    # 학습 시작
    print("Model training start")
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_bar:
            for batch in batch_bar:
                inputs = batch[0].to(device)
                seq_len = inputs.size(1)

                reconstructed, real_or_fake, z, mu, logvar = model(inputs, seq_len)

                recon_loss = criterion(reconstructed, inputs)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_bar.set_postfix(Loss=loss.item())  

        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}")

    print("Saving the model...")
    model_save_path = "./checkpoint/model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
    }, model_save_path)
    print(f"Model saved to {model_save_path}")
