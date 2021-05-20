import os
from math import sqrt
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from VAE import mnist_loader


class VqVae(nn.Module):
    """
    VQ-VAE:
        https://arxiv.org/pdf/1906.00446v1.pdf
    Implementation source:
        https://github.com/ritheshkumar95/pytorch-vqvae
    """
    def __init__(self, hidden_dim=8, latent_vectors=16):
        super().__init__()
        self.encoder = nn.Sequential(
            # Input 28x28
            nn.Conv2d(1, hidden_dim, kernel_size=3, 
                stride=2, padding=1, bias=False),                       # 14x14
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, 
                stride=2, padding=1, bias=False),                       # 7x7
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            # Output 7x7
        )
        self.codebook = VqVaeEmbedding(dim=hidden_dim, k=latent_vectors)
        self.decoder = nn.Sequential(
            # Input 7x7
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, 
                stride=2, padding=1, output_padding=1, bias=False),     # 14x14
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, 1, kernel_size=3, 
                stride=2, padding=1, output_padding=1, bias=False),     # 28x28
            nn.Tanh(),
            # Outpout 28x28
        )
        self.apply(_init)

    @property
    def latent_vectors(self):
        return self.codebook.embedding.shape[1]

    def forward(self, x):
        # Encoding:
        emb = self.encoder(x)

        # Embedding block:
        emb_out, emb_grad, indices = self.codebook.pass_trough(emb)

        # Decoding:
        xhat = self.decoder(emb_out)

        # Returning the reconstruction and both embeddings (before and after the codebook):
        return xhat, emb, emb_grad, indices


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)


class VqVaeEmbedding(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        # Embedding with weights uniformly initialized:
        self.embedding = nn.Embedding(num_embeddings=k, embedding_dim=dim)
        self.embedding.weight.data.uniform_(-1./k, -1./k)

    def forward(self, emb):
        # Returning the plain vector-quantization (no gradient supported):
        emb = emb.permute(0, 2, 3, 1).contiguous()
        return vector_quantization(emb, self.embedding.weight)

    def pass_trough(self, emb):
        # Embedding with gradient support as pass-through:
        emb = emb.permute(0, 2, 3, 1).contiguous()
        emb_out, indices = vector_quantization_pass_through(
            emb, self.embedding.weight.detach())

        # The encoded vectors to be updated by the gradient:
        encoded = self.encode(indices, emb_out.shape)

        # Returning both:
        emb_out = emb_out.permute(0, 3, 1, 2).contiguous()
        return emb_out, encoded, indices

    def encode(self, indices, shape=None):
        if shape is None:
            edge = int(sqrt(len(indices)))
            shape = 1, edge, edge, self.embedding.weight.shape[1]
        return (torch
            .index_select(self.embedding.weight, dim=0, index=indices)
            .view(shape).permute(0, 3, 1, 2).contiguous())



class VectorQuantizationPassThrough(Function):
    """
    This is the full quantization operation, including the gradient pass-through.
    """
    @staticmethod
    def forward(ctx, inputs, codebook):
        # The real quantization:
        inds = vector_quantization(inputs, codebook).view(-1)

        # Storing the data needed for back-propagation:
        ctx.save_for_backward(inds, codebook)
        ctx.mark_non_differentiable(inds)

        # Returning both the vector codes and the associated indices:
        codes = (torch
            .index_select(codebook, dim=0, index=inds)
            .view_as(inputs))
        return (codes, inds)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        # Input gradients are cloned from output:
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()

        # Gradients on the codebook:
        if ctx.needs_input_grad[1]:
            # Data stored during forward pass:
            indices, codebook = ctx.saved_tensors

            # Codebook-sized tensor with gradients in the right places:
            grad_codebook = torch.zeros_like(codebook)
            grad_output_flatten = (grad_output
                .contiguous()
                .view(-1, codebook.size(1)))
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        # Returning both gradients in the right order:
        return (grad_inputs, grad_codebook)


class VectorQuantization(Function):
    """
    The quantization step, given the input:
    - computes the distances wrt the embedding vectors;
    - selects the best ones returning the indicees.
    """    
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            # Flattening to allow distances computation:
            inputs_flatten = inputs.view(-1, codebook.shape[1])

            # Computing \|C\|^2 and \|I\|^2:
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Computing the distances \|I - C\|^2 = \|I\|^2 + \|C\|^2 - 2 I C^T:
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            # Selecting the best vector (index):
            _, indices_flatten = torch.min(distances, dim=1)

            # Back to the input shape:
            indices = indices_flatten.view(*inputs.shape[:-1])
            ctx.mark_non_differentiable(indices)
            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Called `.grad()` on graph containing `VectorQuantization`.')


# Sortcut calls to quantization apply methods:
vector_quantization = VectorQuantization.apply
vector_quantization_pass_through = VectorQuantizationPassThrough.apply


def _init(module):
    """Initialization of conv modules."""
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.fill_(0) # May fail.
        except AttributeError:
            pass


def train(
        model,
        beta=0.25,
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
    loss_fn = vqvae_loss(beta)

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
            xhat, emb, emb_grad, _ = model(x)
            loss = loss_fn(x, xhat, emb, emb_grad)
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
                xhat, emb, emb_grad, _ = model(x)
                test_loss += loss_fn(x, xhat, emb, emb_grad).item()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)

            # Some output:
            print(f"Epoch: {epoch + 1}, Train loss: {train_loss}, Test loss: {test_loss}")

    return model, train_losses, test_losses


def vqvae_loss(beta):
    def the_loss(x, xhat, emb, emb_grad):
        reconstruction = F.mse_loss(xhat, x)
        codebook = F.mse_loss(emb_grad, emb.detach())
        commitment = F.mse_loss(emb, emb_grad.detach())
        return reconstruction + codebook + beta * commitment
    return the_loss


def generate_embeddings(vae, dataset):
    """Preprocessing images from a dataset"""
    data = ((torch.unsqueeze(img,0), label) for img, label in dataset)
    data = ((vae.encoder(tens), label) for tens, label in data)
    data = ((vae.codebook(emb),label) for emb, label in data)
    data = ((torch.flatten(img),label) for img, label in data)
    data = (torch.cat([inds,Tensor([label]).int()]) for inds, label in data)
    return data


def generate_embedding_files(vae, is_train):
    data_loader = mnist_loader(train=is_train)
    data = list(generate_embeddings(vae, data_loader.dataset))
    suffix = 'train' if is_train else 'test'
    torch.save(data, f'generated/vqvae/embedding_{suffix}.pt')
    print(f"Saved {suffix} embeddings")


class EmbeddingDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        data = torch.load(path)
        def process(row):
            img = row.narrow(0,0,row.shape[0]-1)
            label = row.narrow(0,row.shape[0]-1,1)
            return img.reshape((7,7)).contiguous(), label
        self.pairs = [process(row) for row in data]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        return self.pairs[index]
