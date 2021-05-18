import os
import pickle
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as tr
import numpy as np
from matplotlib import pyplot as plt


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


def mnist_loader(train=False, batch_size=16):
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
        post_epoch_every=1,
        batch_size=64, 
        learning_rate=1e-3, 
        train_loader=None,
        test_loader=None,
        checkpoint_path=None,
    ):
    # Data:
    if train_loader is None:
        train_loader = mnist_loader(train=True, batch_size=batch_size)
    if test_loader is None:
        test_loader = mnist_loader(train=False, batch_size=batch_size)

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

        # Not at every epoch:
        if epoch % post_epoch_every == 0:
            # Checkpointing:
            if checkpoint_path is not None:
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), f"{checkpoint_path}/chk{epoch}.pickle")

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


def save_model(path, model, train_losses, test_losses):
    # Saving the model:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path + 'vae.pickle')

    # Saving the losses:
    losses = {
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    with open(path + 'losses.pickle', 'wb') as f:
        pickle.dump(losses, f)


def load_model(path, model):
    # Loading the model:
    model.load_state_dict(torch.load(path + 'vae.pickle'))
    model.eval()

    # Loading the losses:
    with open(path + 'losses.pickle', 'rb') as f:
        losses = pickle.load(f)
        train_losses = losses['train_losses']
        test_losses = losses['test_losses']

    # Return all:
    return model, train_losses, test_losses

def vae_samples_2d(vae, xs=None, ys=None):
    if xs is None:
        xs = np.linspace(-3,3,20)
    if ys is None:
        ys = np.linspace(-3,3,20)
    samples = []
    for y in ys:
        row = []
        for x in xs:
            emb = torch.unsqueeze(torch.tensor([x,y],dtype=torch.float),0)
            row.append(vae.decoder(emb))
        samples.append(row)
    return samples

def plot_vae_samples_2d(samples, cmap=None):
    _, ax = plt.subplots(len(samples[0]), len(samples), figsize=(12,12))
    with torch.no_grad():
        for i in range(len(samples[0])):
            for j in range(len(samples)):
                img = torch.squeeze(samples[i][j], 0)
                ax[i,j].imshow(img.permute(1,2,0), cmap=cmap)
    plt.show()