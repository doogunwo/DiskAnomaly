import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        #print(f"Encoder Input shape: {x.shape}")

        output, (hidden, _) = self.lstm(x)# hx lstm 최종 hidden state
        #print(f"Encoder Hidden state shape: {hidden.shape}") 

        mu = self.hidden_to_mu(hidden[-1])
        logvar = self.hidden_to_logvar(hidden[-1])
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)        # reparameterization trick
        #print(f"Latent z shape: {z.shape}")         # 예상: (batch_size, latent_dim)
        #print(f"Mu shape: {mu.shape}")              # 예상: (batch_size, latent_dim)
        #print(f"Logvar shape: {logvar.shape}")
        return z, mu, logvar

class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        #print(f"Discriminator Input shape: {x.shape}")  # 예상: (batch_size, seq_len, input_dim)

        output, (hidden, _) = self.lstm(x)# hx
        #print(f"Discriminator Hidden state shape: {hidden.shape}")  # 예상: (num_layers, batch_size, hidden_dim)

        last_hidden = hidden[-1]
        output = torch.sigmoid(self.hidden_to_output(last_hidden))
        #print(f"Discriminator Output shape: {output.shape}")  # 예상: (batch_size, 1)

        return output

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(LSTMDecoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z, seq_len):
        batch_size = z.size(0)
        #print(f"Latent z shape: {z.shape}")  # 예상: (batch_size, latent_dim)
        
        hidden_state = self.latent_to_hidden(z).unsqueeze(0)
        cell_state = torch.zeros_like(hidden_state)
        hidden = (hidden_state, cell_state) 

        #print(f"Hidden state shape: {hidden_state.shape}")  # 예상: (1, batch_size, hidden_dim)
        #print(f"Cell state shape: {cell_state.shape}")      # 예상: (1, batch_size, hidden_dim)

        
        input_tensor = torch.zeros(batch_size, seq_len, hidden_state.size(2)).to(z.device)
        #print(f"Input tensor shape: {input_tensor.shape}")
        lstm_output, _ = self.lstm(input_tensor, hidden)
        #print(f"LSTM output shape: {lstm_output.shape}") 
        output = self.output_layer(lstm_output)
        #print(f"Decoder output shape: {output.shape}")
        return output

class AADModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AADModel, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim)
        self.discriminator = LSTMDiscriminator(input_dim, hidden_dim)
        
    def forward(self, x, seq_len):
        z, mu, logvar = self.encoder(x)
        reconstructed = self.decoder(z, seq_len)
        real_or_fake = self.discriminator(reconstructed)
        return reconstructed, real_or_fake, z, mu, logvar
