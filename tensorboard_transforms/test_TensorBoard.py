from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("test_logs")
image_path = "图片路径"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 1, dataformats='HWC')
for i in range(100):
    writer.add_scalar("y = 2x", i, i)

writer.close()