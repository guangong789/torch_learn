import torch
import torchvision
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='CIFAR10下载路径', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.line1 = Linear(196608, 10)

    def forward(self, input):
        output = self.line1(input)
        return output


network = Network()

for data in dataloader:
    img, label = data
    print(img.shape)
    output = torch.flatten(img)
    print(output.shape)
    output = network(output)
    print(output.shape)