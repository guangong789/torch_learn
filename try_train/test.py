import torch
import torchvision
from PIL import Image
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_path = "测试图片路径"
image = Image.open(img_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


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


model = torch.load("./cifar10_models/cifar10_net_25.pth")  # 已保存的模型的路径
model = model.to(device)
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
image = image.to(device)

model.eval()
with torch.no_grad():
    output = model(image)
print(output)

species = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = output.argmax(1)
print(species[index.item()])