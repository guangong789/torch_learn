from datetime import time

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 设置训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = torchvision.datasets.CIFAR10(root='CIFAR10下载的路径', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='CIFAR10下载的路径', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为: {}".format(train_data_size))
print("测试数据集长度为: {}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 创建网络模型
net = Net()
net = net.to(device)

# 定义损失函数
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(device)

# 定义优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(), learning_rate)

# 设置训练网络的参数
total_train_step = 0  # 记录训练的总步数
total_test_step = 0  # 记录测试的总步数
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter(log_dir='cifar10')
for i in range(epoch):
    print(f"________第{i + 1}轮训练开始________")
    # 训练开始
    net.train()
    # 获取训练数据
    for data in train_dataloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = net(images)
        # 计算损失
        loss = loss_func(outputs, labels)

        # 优化器梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # 记录训练的总步数
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}, Loss：{loss.item()}")
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    net.eval()
    # 测试开始
    total_test_loss = 0  # 记录测试的总损失
    total_accuracy = 0  # 记录测试的总准确率
    with torch.no_grad():
        # 获取测试数据
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播
            outputs = net(images)
            # 计算损失
            loss = loss_func(outputs, labels)
            # 记录测试的总步数
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(dim=1) == labels).sum()
            total_accuracy += accuracy.item()

        print(f"整体测试集上的Loss：{total_test_loss}")
        print(f"整体测试集上的准确率：{total_accuracy / test_data_size}")
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
        total_test_step += 1

        torch.save(net, f"cifar10_net_{i + 1}.pth")
        # torch.save(net.state_dict(), f"cifar10_net_{i+1}.pth")
        print(f"模型保存成功！")

writer.close()