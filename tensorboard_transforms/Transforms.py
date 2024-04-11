from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# tensor数据类型

img_path = "图片路径"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()  # 创建ToTensor类工具 tensor_trans

tensor_img = tensor_trans(img)  # 使用工具 tensor_img

writer.add_image("tensor_img", tensor_img)

writer.close()