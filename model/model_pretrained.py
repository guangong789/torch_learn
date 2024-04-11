import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights='DEFAULT')

train_data = torchvision.datasets.CIFAR10(root='CIFAR10下载路径', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

print(vgg16_true)
# 增加一个全连接层
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
# 修改一个全连接层
vgg16_false.classifier.add_module('6', nn.Linear(4096, 10))
print(vgg16_false)