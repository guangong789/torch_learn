import torch
from torch import nn
from torch.nn import MSELoss
from torch.nn.functional import l1_loss

inputs = torch.tensor([[1, 2, 3]], dtype=torch.float32)
targets = torch.tensor([[1, 2, 5]], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

result_l1 = l1_loss(inputs, targets, reduction='sum')
print(result_l1)

mse_loss = MSELoss()
result_mse = mse_loss(inputs, targets)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))

loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)