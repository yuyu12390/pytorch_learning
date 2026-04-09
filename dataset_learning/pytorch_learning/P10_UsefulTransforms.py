from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../logs")

img = Image.open("../image/terminal.png")

#ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

#Normalize
print(img_tensor[0][0][0])
trans_normalize = transforms.Normalize([1, 3, 5],[3, 2, 1])
img_normalize = trans_normalize(img_tensor)
print(img_normalize[0][0][0])
writer.add_image("toNormalize",img_normalize,1)

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
print(img_resize)
writer.add_image("toResize",img_resize,1)

#compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("toResize",img_resize_2,2)

#RandomCrop
trans_random = transforms.RandomCrop((500,800))
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("toCrop",img_crop,i)

print(trans_random.shape)

writer.close()