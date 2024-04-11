import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="CIFAR10的下载路径", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="CIFAR10的下载路径", train=False, transform=dataset_transform, download=True)
# 原始图片类型: PIL.image -> 4~6行 -> tensor类型

print(test_set[0])
print(test_set.classes)

img, target = test_set[0]
print(img)
print(target)
print(test_set.classes[target])
img.show()

print(test_set[0])

writer = SummaryWriter("CIFAR10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()