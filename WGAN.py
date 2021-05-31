import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from VAE import mnist_loader, Reshape
from VQ_VAE import _init
from GAN import model_parameters


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


class Critic(nn.Sequential):
    def __init__(self, c=4):
        super().__init__(
            *self._conv_norm_relu(1, c),
            *self._conv_norm_relu(c, c*2),
            *self._conv_norm_relu(c*2, c*4),
            nn.Conv2d(c*4, c*8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Flatten(),
            nn.Linear((c*8)*(2*2), 1),
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
        lambda_gp=10,
        epochs=20, 
        post_epoch_every=1,
        batch_size=64, 
        critic_iterations=1,
        lr=1e-3, 
        train_loader=None,
        test_loader=None,
        checkpoint_path=None,
    ):
    # The models:
    generator = Generator(hidden_dims)
    critic = Critic()
    print(f"Generator parameters: {model_parameters(generator)}")
    print(f"Critic parameters: {model_parameters(critic)}")

    # The data:
    if train_loader is None:
        train_loader = mnist_loader(
            train=True, 
            batch_size=batch_size,
        )
    if test_loader is None:
        test_loader = mnist_loader(train=False, batch_size=batch_size)

    # The optimization:
    optimizer_generator = optim.Adam(generator.parameters(), lr=lr, betas=(0.0001, 0.9))
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0001, 0.9))

    # The losses:
    def critic_loss_fn(xreal, xfake):
        critic_loss = -(torch.mean(critic(xreal)) - torch.mean(critic(xfake)))
        gp_loss = gradient_penalty(xreal, xfake, critic)
        #print(f"gp: {gp_loss.item()}")
        #print(f"cl: {critic_loss.item()}")
        return critic_loss + lambda_gp*gp_loss
    def generator_loss_fn(xfake):
        return -torch.mean(critic(xfake))

    # Noise for output fakes generation:
    noise = torch.rand((1,hidden_dims))

    # Training:
    train_generator_losses = []
    train_critic_losses = []
    test_critic_losses = []
    fakes = []
    for epoch in range(epochs):
        # One pass on the data:
        train_generator_loss = []
        train_critic_loss = []
        for i, (xreal, _) in enumerate(train_loader):
            # Initialization of the critic optimization step:
            critic.train()
            optimizer_critic.zero_grad()

            # Fake samples:
            n = xreal.shape[0]
            xfake = generator(torch.rand((n,hidden_dims)))

            # The loss:
            critic_loss = critic_loss_fn(xreal, xfake)
            train_critic_loss.append(critic_loss.item())

            # Optimization (critic):
            critic_loss.backward()
            optimizer_critic.step()

            # Do not move the generator at every batch:
            if (i+1) % critic_iterations == 0:
                # Initialization of the generator optimization step:
                generator.train()
                optimizer_generator.zero_grad()

                # The loss:
                xfake = generator(torch.rand((n,hidden_dims)))
                generator_loss = generator_loss_fn(xfake)
                #print(f"Gl: {generator_loss.item()}")
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
        train_critic_losses.append(np.mean(train_critic_loss))
        train_generator_losses.append(np.mean(train_generator_loss))

        # Not at every epoch:
        if epoch % post_epoch_every == 0:
            # Checkpointing:
            if checkpoint_path is not None:
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(generator.state_dict(), f"{checkpoint_path}/chk_generator_{epoch}.pickle")
                torch.save(critic.state_dict(), f"{checkpoint_path}/chk_critic_{epoch}.pickle")

            # Evaluating on the test:
            critic.eval()
            test_loss = []
            for xreal, _ in test_loader:
                # Fake samples:
                n = xreal.shape[0]
                xfake = generator(torch.rand((n,hidden_dims)))

                # The loss:
                critic_loss = critic_loss_fn(xreal, xfake)
                test_loss.append(critic_loss.item())
            test_critic_losses.append(np.mean(test_loss))

            # Some output:
            print(f"Epoch: {epoch + 1}, losses (G,Ctr,Cte): " + 
                f"{train_generator_losses[-1]}, {train_critic_losses[-1]}, {test_critic_losses[-1]}")

    # Completed:
    return generator, critic, fakes, train_generator_losses, train_critic_losses, test_critic_losses


def gradient_penalty(xreal, xfake, critic):
    # Interpolating real and fake images:
    n, c, h, w = xreal.shape
    alpha = torch.rand((n, 1, 1, 1))
    alpha = alpha.repeat((1, c, h, w))
    xmix = alpha * xreal + (1-alpha) * xfake

    # Whole model evaluation (forward pass):
    critic_xmix = critic(xmix)

    # Gradient in xmix (backward pass):
    grad_xhat = torch.autograd.grad(
        inputs=xmix,
        outputs=critic_xmix,
        grad_outputs=torch.ones_like(critic_xmix),
        create_graph=True,
    )[0]
    grad_xhat = grad_xhat.view((grad_xhat.shape[0], -1))

    # Returning its norm - 1 squared:
    return torch.mean((grad_xhat.norm(2, dim=1) - 1)**2)
