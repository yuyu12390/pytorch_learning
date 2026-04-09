from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = "../dataset/train/ants_image/0013035.jpg"  #相对路径
#绝对路径 img_path_abs = "D:\py\dataset_learning\dataset\train\ants_image\0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("../logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("tensor_img", tensor_img)
print(tensor_img)
