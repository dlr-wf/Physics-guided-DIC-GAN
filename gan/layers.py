import torch
from torch import nn, Tensor


class Reshape(nn.Module):
    """
        Costum reshaping layer similar to Tensorflows Reshape layer
    """
    def __init__(self, dim: tuple) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.reshape(input=x, shape=self.dim)
