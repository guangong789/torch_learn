import torch
import torchvision

# 加载方式一 -> 保存方式一，加载模型
model1 = torch.load("vgg16_method1.pth")
print(model1)

# 加载方式二 -> 保存方式二，加载模型
vgg16 = torchvision.models.vgg16(weights='DEFAULT')
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load("vgg16_method2.pth")
# print(model2)
print(vgg16)
