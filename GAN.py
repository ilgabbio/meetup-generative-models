import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from VAE import mnist_loader, Reshape
from VQ_VAE import _init


class Generator(nn.Sequential):
    def __init__(self, c=16):
        super().__init__(
            # Input 8
            Reshape((-1,2,2,2)),
            nn.ConvTranspose2d(2, c, kernel_size=3, stride=2, 
                padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(c, c//2, kernel_size=3, stride=2, 
                padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(c//2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(c//2, c//4, kernel_size=3, stride=2, 
                padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(c//4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(c//4, 1, kernel_size=3, stride=2, 
                padding=1, output_padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Tanh(),
            # Output 1x28x28
        )
        _init(self)


class Discriminator(nn.Sequential):
    def __init__(self, c=2):
        super().__init__(
            # Input 1x28x28
            nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c*2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c*2, c*4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c*4, c*8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
            nn.Linear((c*8)*(2*2), 1),
            nn.Sigmoid(),
            # Output 1
        )
        _init(self)

def train(
        epochs=20, 
        post_epoch_every=1,
        batch_size=64, 
        critic_iterations=1,
        lr_generator=3e-4, 
        lr_discriminator=3e-4, 
        train_loader=None,
        test_loader=None,
        checkpoint_path=None,
    ):
    # The models:
    generator = Generator()
    discriminator = Discriminator()
    print(f"Generator parameters: {model_parameters(generator)}")
    print(f"Discriminator parameters: {model_parameters(discriminator)}")

    # The data:
    if train_loader is None:
        train_loader = mnist_loader(
            train=True, 
            batch_size=batch_size,
            #sampler=SubsetRandomSampler(range(640))
        )
    if test_loader is None:
        test_loader = mnist_loader(train=False, batch_size=batch_size)

    # The optimization:
    optimizer_generator = optim.Adam(generator.parameters(), lr=lr_generator, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    # Noise for output fakes generation:
    noise = torch.rand((1,8))

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
            # Many discriminator steps:
            for _ in range(critic_iterations):
                # Initialization of the discriminator optimization step:
                discriminator.train()
                optimizer_discriminator.zero_grad()

                # Fake samples:
                n = xreal.shape[0]
                xfake = generator(torch.rand((n,8)))

                # The labels:
                real = torch.full((n,1), 1, dtype=torch.float)
                fake = torch.full((n,1), 0, dtype=torch.float)

                # The loss:
                real_loss = adversarial_loss(discriminator(xreal), real)
                fake_loss = adversarial_loss(discriminator(xfake), fake)
                discriminator_loss = (real_loss + fake_loss) / 2
                train_discriminator_loss.append(discriminator_loss.item())

                # Optimization (discriminator):
                discriminator_loss.backward()
                optimizer_discriminator.step()

            # Initialization of the generator optimization step:
            generator.train()
            optimizer_generator.zero_grad()

            # The loss:
            xfake = generator(torch.rand((n,8)))
            generator_loss = adversarial_loss(discriminator(xfake), real) # Fooling
            train_generator_loss.append(generator_loss.item())

            # Optimization (generator):
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
                xfake = generator(torch.rand((n,8)))

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
