import torch
from torch import nn


class Network(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


network = Network()
x = torch.tensor(1.0)
output = network(x)
print(output)
