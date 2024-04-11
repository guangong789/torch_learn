import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class NNSeq(nn.Module):
    def __init__(self):
        super(NNSeq, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2, stride=1),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


seq_network = NNSeq()
print(seq_network)
input = torch.ones((64, 3, 32, 32))
output = seq_network(input)
print(output.shape)

writer = SummaryWriter("nn_seq")
writer.add_graph(seq_network, input)

writer.close()