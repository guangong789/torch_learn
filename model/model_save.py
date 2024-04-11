import torch
import torchvision

vgg16 = torchvision.models.vgg16(weights=None)
# 保存方式一(模型结构和参数)
torch.save(vgg16, 'vgg16_method1.pth')

# 保存方式二(仅模型参数,官方推荐)
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')
