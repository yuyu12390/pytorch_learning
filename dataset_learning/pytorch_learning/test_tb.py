from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

Writer = SummaryWriter("../logs")
img_path = "/dataset/train/bees_image/17209602_fe5a5a746f.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

print(type(img_array))

Writer.add_image("test", img_array,2,dataformats="HWC")

for i in range(100):
    Writer.add_scalar("y=2x",3*i,i)

Writer.close()