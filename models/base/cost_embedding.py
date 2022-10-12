import torch.nn as nn
from models.base.conv4d import Encoder4D, Encoder_T_4D

class CostEmbedding4d_64(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedder = Encoder4D( # Encoder for conv_5
                corr_levels=(1, 4, 16),
                kernel_size=(
                    (3, 3, 3, 3),
                    (3, 3, 3, 3)
                ),
                stride=(
                    (2, 2, 2, 2),
                    (2, 2, 2, 2)
                ),
                padding=(
                    (1, 1, 1, 1),
                    (1, 1, 1, 1)
                ),
                group=(1, 1),
            )
    def forward(self, x):
        
        x = x.view(-1, 1, 64, 64, 64, 64)
  
        return self.embedder(x)

class CostEmbedding4d_32(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedder = Encoder4D( # Encoder for conv_5
                corr_levels=(1, 4, 16),
                kernel_size=(
                    (3, 3, 3, 3),
                    (3, 3, 3, 3)
                ),
                stride=(
                    (2, 2, 2, 2),
                    (1, 1, 1, 1)
                ),
                padding=(
                    (1, 1, 1, 1),
                    (1, 1, 1, 1)
                ),
                group=(1, 1),
            )
    def forward(self, x):

        x = x.view(-1, 1, 32, 32, 32, 32)

        return self.embedder(x)

class CostEmbedding4d_16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedder = Encoder4D( # Encoder for conv_5
                corr_levels=(1, 4, 16),
                kernel_size=(
                    (3, 3, 3, 3),
                    (3, 3, 3, 3)
                ),
                stride=(
                    (1, 1, 1, 1),
                    (1, 1, 1, 1)
                ),
                padding=(
                    (1, 1, 1, 1),
                    (1, 1, 1, 1)
                ),
                group=(1, 1),
            )
    def forward(self, x):

        x = x.view(-1, 1, 16, 16, 16, 16)
      
        return self.embedder(x)

class CostEmbedding4d_8(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedder = Encoder_T_4D( # Encoder for conv_5
                corr_levels=(1, 4, 16),
                kernel_size=(
                    (3, 3, 3, 3),
                    (3, 3, 3, 3)
                ),
                stride=(
                    (2, 2, 2, 2), 
                    (1, 1, 1, 1)
                ),
                padding=(
                    (1, 1, 1, 1),
                    (1, 1, 1, 1),
                ),
                out_padding=(
                    (1, 1, 1, 1),
                    (0, 0, 0, 0)
                ),
                target_size =(
                    (16, 16, 16, 16),
                    (16, 16, 16, 16)
                ),
                group=(1, 1),
            )
    def forward(self, x):

        x = x.view(-1, 1, 8, 8, 8, 8)

        return self.embedder(x)

class CostEmbedding4d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedder = Encoder4D( # Encoder for conv_5
                corr_levels=(65, 64),
                kernel_size=(
                    (3, 3, 3, 3),
                ),
                stride=(
                    (1, 1, 1, 1),
                ),
                padding=(
                    (1, 1, 1, 1),
                ),
                group=(4, 4),
            )
    def forward(self, x):

        return self.embedder(x)

