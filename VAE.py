import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as tr

## The model

class DownscaleConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=3, padding=padding),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


class UpscaleConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=3, padding=padding),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
        )


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return torch.reshape(x, self.shape)


class Vae(nn.Module):
    def __init__(self, hidden_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            # Input is 28x28.
            DownscaleConv2d(1, 16),                 # 14x14
            DownscaleConv2d(16, 8),                 # 7x7
            DownscaleConv2d(8, 8, padding=2),       # 4x4
            nn.Flatten(),
            # Result is flat with 8x4x4 values.
        )
        self.mu = nn.Linear(8*4*4, hidden_dim)
        self.log_var = nn.Linear(8*4*4, hidden_dim)
        self.decoder = nn.Sequential(
            # Here we receive the gaussian sample.
            nn.Linear(hidden_dim, 8*4*4),
            Reshape((-1,8,4,4)),                # 4x4
            UpscaleConv2d(8, 8),                # 8x8
            UpscaleConv2d(8, 16),               # 16x16
            UpscaleConv2d(16, 32, padding=0),   # 28x28
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
            # Returning an image with just one channel in [0-1].
            nn.Sigmoid()
        )

    def forward(self, x):
        # Estimated parameters (encoding):
        emb = self.encoder(x)
        mu = self.mu(emb)
        log_var = self.log_var(emb)
        std = torch.exp(0.5*log_var)

        # The stochastic part:
        eps = torch.randn((mu.shape[0],self.hidden_dim))
        h = eps * std + mu

        # Decoding the image:
        return self.decoder(h), mu, log_var


def loader(train=False, batch_size=16):
    return DataLoader(
        MNIST(
            'generated/data/',
            train=train,
            download=True,
            transform=tr.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True)


def vae_loss(beta=1):
    def the_loss(x, xhat, mean, log_var):
        reconstruction_error = F.binary_cross_entropy(xhat,x,reduction='sum')
        # reconstruction_error = F.mse_loss(xhat,x,reduction='sum')
        # See eq.40 in the ref.
        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # beta-VEA: https://openreview.net/references/pdf?id=Sy2fzU9gl
        return reconstruction_error + beta*kl_divergence
    return the_loss


def train(
        model, 
        beta=1,
        epochs=20, 
        batch_size=64, 
        learning_rate=1e-3, 
        checkpoint_path=None
    ):
    # Data:
    train_loader = loader(train=True, batch_size=batch_size)
    test_loader = loader(train=False, batch_size=batch_size)

    # The optimization:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = vae_loss(beta=beta)

    # Training:
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        # One whole epoch:
        train_loss = 0
        for x, _ in train_loader:
            # One optimization step:
            model.train()
            optimizer.zero_grad()
            xhat, mu, log_var = model(x)
            loss = loss_fn(x, xhat, mu, log_var)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Checkpointing:
        if checkpoint_path is not None:
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(), f"{checkpoint_path}/chk{epoch}.pickle");

        # Evaluating on the test:
        model.eval()
        test_loss = 0
        for x, _ in test_loader:
            xhat, mu, log_var = model(x)
            test_loss += loss_fn(x, xhat, mu, log_var).item()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        # Some output:
        print(f"Epoch: {epoch + 1}, Train loss: {train_loss}, Test loss: {test_loss}")

    return model, train_losses, test_losses
