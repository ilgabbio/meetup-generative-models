import os
import torch
from torch._C import dtype
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from VQ_VAE import EmbeddingDataset, _init


class GatedPixelCNN(nn.Module):
    def __init__(self, 
            hidden_dim=8, 
            latent_vectors=16, 
            n_layers=15, 
            final_channels=64, 
            n_classes=10
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_vectors = latent_vectors

        # Embedding layer to learn code-specific latents:
        self.embedding = nn.Embedding(latent_vectors, hidden_dim)

        # Building the PixelCNN layer by layer:
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Initial block with Mask-A convolution, rest with Mask-B:
            if i== 0:
                self.layers.append(
                    GatedMaskedConv2d('A', hidden_dim, kernel=7, 
                        residual=False, n_classes=n_classes)
                )
            else:
                self.layers.append(
                    GatedMaskedConv2d('B', hidden_dim, kernel=3, 
                        residual=True, n_classes=n_classes)
                )

        # Add the output layer:
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(True),
            nn.Conv2d(final_channels, latent_vectors, kernel_size=1)
        )

        # Initializing convolutions:
        self.apply(_init)

    def forward(self, x, label):
        # Mapping every code via embedding:
        x = (self
            .embedding(x.view(-1))
            .view(x.size() + (-1,)) # (B, H, W, C)
            .permute(0, 3, 1, 2))   # (B, C, W, W)

        # All the gated convolutional layers:
        x_v, x_h = (x, x)
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h, label)

        # Final convolutions to get the output with correct size:
        return self.output_conv(x_h)

    def generate(self, label, shape=(7, 7), batch_size=9):
        # Base input:
        if not torch.is_tensor(label):
            label = torch.Tensor(label).int()
        x = torch.zeros((batch_size, *shape), dtype=torch.int32)

        # Generating the single pixels in the image (scan):
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Estimating the distributions:
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)

                # The winning class becomes the value:
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        
        # Generated batch of images:
        return x


class GatedMaskedConv2d(nn.Module):
    """
    Gated-masked convolution with conditional regolarization.
    See:
     - paoer https://papers.nips.cc/paper/2016/file/b1301141feffabac455e1f90a7de2054-Paper.pdf
     - figure 2 for the schema
     - equation 4 for the gate activation
    """
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        # Embedding vectors, one per class, 2 dim parts (for the v|h split):
        self.class_cond_embedding = nn.Embedding(n_classes, 2 * dim)

        # The convolution connections:
        self.vert_stack = nn.Sequential(
            nn.Conv2d(
                dim, dim * 2,
                kernel_size=(kernel // 2 + 1, kernel),  # (ceil(n/2), n)
                bias=False,
                stride=1, 
                padding=(kernel // 2, kernel // 2)
            ),
            nn.BatchNorm2d(dim * 2)
        )
        self.vert_to_horiz = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, 1, bias=False),
            nn.BatchNorm2d(dim * 2)
        )
        self.horiz_stack = nn.Sequential(
            nn.Conv2d(
                dim, dim * 2,
                kernel_size=(1, kernel // 2 + 1),
                bias=False,
                stride=1, 
                padding=(0, kernel // 2)
            ),
            nn.BatchNorm2d(dim * 2)
        )
        self.horiz_resid = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # A gate for the activations:
        self.gate = GatedActivation()

    def forward(self, x_v, x_h, label):
        if self.mask_type == 'A':
            self.make_causal()

        # Selecting in the embedding:
        if len(label.shape) > 1:
            label = torch.squeeze(label,1)
        h = self.class_cond_embedding(label)

        # Vertical-stack convolution and gate activation:
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        # Vertical-stack convolution and gate activation:
        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)
        out = self.gate(v2h + h_horiz + h[:, :, None, None])

        # Optionally adding the residual connection:
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        # Output next version of v and h:
        return out_v, out_h

    def make_causal(self):
        """Mask final row on vert and final column on horiz."""
        self.vert_stack[0].weight.data[:, :, -1].zero_()
        self.horiz_stack[0].weight.data[:, :, :, -1].zero_()


class GatedActivation(nn.Module):
    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


def train(
        model,
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
        ds_train = EmbeddingDataset('generated/vqvae/embedding_train.pt')
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    if test_loader is None:
        ds_test = EmbeddingDataset('generated/vqvae/embedding_test.pt')
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    # The optimization:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = lambda logits, latents: F.cross_entropy(
        logits.permute(0, 2, 3, 1).contiguous().view(-1, model.latent_vectors),
        latents.view(-1))

    # Training:
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        # One whole epoch:
        train_loss = 0
        for encoded, label in train_loader:
            # One optimization step:
            model.train()
            optimizer.zero_grad()
            logits = model(encoded, label)
            loss = loss_fn(logits, encoded)
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
            for encoded, label in test_loader:
                logits = model(encoded, label)
                test_loss += loss_fn(logits, encoded).item()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)

            # Some output:
            print(f"Epoch: {epoch + 1}, Train loss: {train_loss}, Test loss: {test_loss}")

    return model, train_losses, test_losses