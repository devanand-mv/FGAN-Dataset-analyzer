import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class TabularGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128):
        super(TabularGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output range in [-1, 1] matching preprocessed minmax scaling
        )

    def forward(self, z):
        return self.net(z)


class TabularDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(TabularDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


class TabularGAN:
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128, lr=0.0002, device=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.generator = TabularGenerator(latent_dim, input_dim, hidden_dim).to(self.device)
        self.discriminator = TabularDiscriminator(input_dim, hidden_dim).to(self.device)
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = {"epoch": [], "d_loss": [], "g_loss": []}

    def train(self, data_numpy, epochs=150, batch_size=32, progress_callback=None):
        """
        Trains the Generator and Discriminator on real minority-class tabular samples.
        """
        self.generator.train()
        self.discriminator.train()
        
        tensor_data = torch.tensor(data_numpy, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        
        # Adjust batch size if data is smaller
        actual_batch_size = min(batch_size, max(len(data_numpy) // 2, 8))
        dataloader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, drop_last=False)
        
        self.history = {"epoch": [], "d_loss": [], "g_loss": []}
        
        for epoch in range(1, epochs + 1):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            batches = 0
            
            for (real_batch,) in dataloader:
                b_size = real_batch.size(0)
                if b_size <= 1:
                    continue
                real_batch = real_batch.to(self.device)
                
                # --- Train Discriminator ---
                self.optimizer_D.zero_grad()
                
                # One-sided label smoothing for real: 0.9 instead of 1.0
                real_labels = torch.full((b_size, 1), 0.9, device=self.device)
                fake_labels = torch.zeros((b_size, 1), device=self.device)
                
                # Real pass
                d_out_real = self.discriminator(real_batch)
                d_loss_real = self.criterion(d_out_real, real_labels)
                
                # Fake pass
                z = torch.randn(b_size, self.latent_dim, device=self.device)
                fake_batch = self.generator(z)
                d_out_fake = self.discriminator(fake_batch.detach())
                d_loss_fake = self.criterion(d_out_fake, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer_D.step()
                
                # --- Train Generator ---
                self.optimizer_G.zero_grad()
                z = torch.randn(b_size, self.latent_dim, device=self.device)
                fake_batch = self.generator(z)
                d_out_fake_for_g = self.discriminator(fake_batch)
                # Generator wants discriminator to believe fakes are real
                g_loss = self.criterion(d_out_fake_for_g, torch.ones((b_size, 1), device=self.device))
                
                g_loss.backward()
                self.optimizer_G.step()
                
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                batches += 1
                
            if batches > 0:
                avg_d_loss = epoch_d_loss / batches
                avg_g_loss = epoch_g_loss / batches
            else:
                avg_d_loss = 0.0
                avg_g_loss = 0.0
                
            self.history["epoch"].append(epoch)
            self.history["d_loss"].append(round(avg_d_loss, 4))
            self.history["g_loss"].append(round(avg_g_loss, 4))
            
            if progress_callback is not None:
                progress_callback(epoch, epochs, avg_d_loss, avg_g_loss)
                
        return self.history

    def generate(self, n_samples):
        """Generates synthetic tabular features from latent Gaussian noise."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            synthetic = self.generator(z).cpu().numpy()
        return synthetic
