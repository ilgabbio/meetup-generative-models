import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from VAE import mnist_loader, Reshape
from VQ_VAE import _init


class Generator(nn.Sequential):
    def __init__(self, hidden_dims=16, c=16):
        super().__init__(
            Reshape((-1,hidden_dims,1,1)),
            *self._conv_norm_relu(hidden_dims, c*2),
            *self._conv_norm_relu(c*2, c),
            *self._conv_norm_relu(c, c//2),
            *self._conv_norm_relu(c//2, c//4, padding=2),
            *self._conv_norm_relu(c//4, c//8),
            nn.Conv2d(c//8, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        _init(self)

    def _conv_norm_relu(self, ch_in, ch_out, padding=1):
        return [
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=2, 
                padding=padding, output_padding=1, bias=False),
            nn.InstanceNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
        ]


class Discriminator(nn.Sequential):
    def __init__(self, c=4):
        super().__init__(
            *self._conv_norm_relu(1, c),
            *self._conv_norm_relu(c, c*2),
            *self._conv_norm_relu(c*2, c*4),
            nn.Conv2d(c*4, c*8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Flatten(),
            nn.Linear((c*8)*(2*2), 1),
            nn.Sigmoid(),
        )
        _init(self)

    def _conv_norm_relu(self, ch_in, ch_out):
        return [
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ch_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

def train(
        hidden_dims=16,
        epochs=20, 
        post_epoch_every=1,
        batch_size=64, 
        lr_generator=3e-4, 
        lr_discriminator=3e-4, 
        train_loader=None,
        test_loader=None,
        checkpoint_path=None,
    ):
    # The models:
    generator = Generator(hidden_dims=hidden_dims)
    discriminator = Discriminator()
    print(f"Generator parameters: {model_parameters(generator)}")
    print(f"Discriminator parameters: {model_parameters(discriminator)}")

    # The data:
    if train_loader is None:
        train_loader = mnist_loader(
            train=True, 
            batch_size=batch_size,
        )
    if test_loader is None:
        test_loader = mnist_loader(train=False, batch_size=batch_size)

    # The optimization:
    optimizer_generator = optim.Adam(generator.parameters(), lr=lr_generator, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    # Noise for output fakes generation:
    noise = torch.rand((1,hidden_dims))

    # Training:
    train_generator_losses = []
    train_discriminator_losses = []
    test_discriminator_losses = []
    fakes = []
    for epoch in range(epochs):
        # One pass on the data:
        train_generator_loss = []
        train_discriminator_loss = []
        for xreal, _ in train_loader:
            # Discriminator
            # -------------

            # Initialization of the discriminator optimization step:
            discriminator.train()

            # Fake samples:
            n = xreal.shape[0]
            xfake = generator(torch.rand((n,hidden_dims)))

            # The labels:
            real = torch.full((n,1), 1, dtype=torch.float)
            fake = torch.full((n,1), 0, dtype=torch.float)

            # The loss:
            real_loss = adversarial_loss(discriminator(xreal), real)
            fake_loss = adversarial_loss(discriminator(xfake), fake)
            discriminator_loss = (real_loss + fake_loss) / 2
            train_discriminator_loss.append(discriminator_loss.item())

            # Optimization (discriminator):
            optimizer_discriminator.zero_grad()
            discriminator_loss.backward()
            optimizer_discriminator.step()

            # Generator
            # ---------

            # Initialization of the generator optimization step:
            generator.train()

            # The loss:
            xfake = generator(torch.rand((n,hidden_dims)))
            generator_loss = adversarial_loss(discriminator(xfake), real) # Fooling
            train_generator_loss.append(generator_loss.item())

            # Optimization (generator):
            optimizer_generator.zero_grad()
            generator_loss.backward()
            optimizer_generator.step()
        
        # Generating a fake:
        img = generator(noise)[0].permute(1,2,0).detach()
        fakes.append(img)
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(img)
        plt.show()

        # Epoch losses:
        train_discriminator_losses.append(np.mean(train_discriminator_loss))
        train_generator_losses.append(np.mean(train_generator_loss))

        # Not at every epoch:
        if epoch % post_epoch_every == 0:
            # Checkpointing:
            if checkpoint_path is not None:
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(generator.state_dict(), f"{checkpoint_path}/chk_generator_{epoch}.pickle")
                torch.save(discriminator.state_dict(), f"{checkpoint_path}/chk_discriminator_{epoch}.pickle")

            # Evaluating on the test:
            discriminator.eval()
            test_loss = []
            for xreal, _ in test_loader:
                # Fake samples:
                n = xreal.shape[0]
                xfake = generator(torch.rand((n,hidden_dims)))

                # The labels:
                real = torch.full((n,1), 1, dtype=torch.float)
                fake = torch.full((n,1), 0, dtype=torch.float)

                # The loss:
                real_loss = adversarial_loss(discriminator(xreal), real)
                fake_loss = adversarial_loss(discriminator(xfake), fake)
                discriminator_loss = (real_loss + fake_loss) / 2
                test_loss.append(discriminator_loss.item())
            test_discriminator_losses.append(np.mean(test_loss))

            # Some output:
            print(f"Epoch: {epoch + 1}, losses (G,Dtr,Dte): {train_generator_losses[-1]}, {train_discriminator_losses[-1]}, {test_discriminator_losses[-1]}, ")

    # Completed:
    return generator, discriminator, fakes, train_generator_losses, train_discriminator_losses, test_discriminator_losses

def model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def save_model(path, gen, disc, fakes, losses):
    # Saving the model:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(gen.state_dict(), path + 'gen.pickle')
    torch.save(disc.state_dict(), path + 'disc.pickle')

    # The fakes:
    with open(path + 'fakes.pickle', 'wb') as f:
        pickle.dump(fakes, f)

    # Saving the losses:
    with open(path + 'losses.pickle', 'wb') as f:
        pickle.dump(losses, f)

def load_model(path, gen, disc):
    # Loading the model:
    gen.load_state_dict(torch.load(path + 'gen.pickle'))
    disc.load_state_dict(torch.load(path + 'disc.pickle'))
    gen.eval()
    disc.eval()

    # The fakes:
    with open(path + 'fakes.pickle', 'rb') as f:
        fakes = pickle.load(f)

    # Loading the losses:
    with open(path + 'losses.pickle', 'rb') as f:
        losses = pickle.load(f)

    # Return all:
    return gen, disc, fakes, losses
