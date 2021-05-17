import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchvision.transforms as tr
from VAE import DownscaleConv2d, UpscaleConv2d, Reshape, Vae


class FacesDataset(Dataset):
    def __init__(self, 
                 path='data/face_data.csv', 
                 subind=None,
                 transform=tr.ToTensor()
            ):
        super().__init__()
        self.path = path
        faces = pd.read_csv(path).sample(frac=1)
        faces = faces.drop('target',axis=1)
        self.faces = np.array(faces)
        if subind is not None:
            self.faces = self.faces[subind]
        self.transform = transform

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, index):
        # The index may be a tensor:
        if torch.is_tensor(index):
            index = index.tolist()

        # The face as an image:
        face = self.faces[index].reshape(64,64,1)
        face = face.astype(np.float32)

        # Transformations:
        if self.transform is not None:
            face = self.transform(face)

        # Done:
        return (face,0) # Second place for labels.


def faces_loader(train=False, batch_size=16):
    subind = range(0,350) if train else range(350,400)
    return DataLoader(
        dataset=FacesDataset(subind=subind),
        batch_size=batch_size,
        shuffle=True
    )


class VaeFaces(Vae):
    def __init__(self, hidden_dim=2):
        super().__init__(hidden_dim)
        self.encoder = nn.Sequential(
            # Input is 64x64.
            DownscaleConv2d(1, 16),     # 32x32
            DownscaleConv2d(16, 8),     # 16x16
            DownscaleConv2d(8, 4),      # 8x8
            DownscaleConv2d(4, 2),      # 4x4
            DownscaleConv2d(2, 1),      # 2x2
            nn.Flatten(),
            # Result is flat with 1x2x2 values.
        )
        self.mu = nn.Linear(1*2*2, hidden_dim)
        self.log_var = nn.Linear(1*2*2, hidden_dim)
        self.decoder = nn.Sequential(
            # Here we receive the gaussian sample.
            nn.Linear(hidden_dim, 1*2*2),
            Reshape((-1,1,2,2)),        # 2x2
            UpscaleConv2d(1, 2),        # 4x4
            UpscaleConv2d(2, 4),        # 8x8
            UpscaleConv2d(4, 8),        # 16x16
            UpscaleConv2d(8, 16),       # 32x32
            UpscaleConv2d(16, 32),       # 64x64
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
            # Returning an image with just one channel in [0-1].
            nn.Sigmoid()
        )

