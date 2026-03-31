import torchvision
from torch.utils.tensorboard import SummaryWriter
dataset_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])

train_set = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, transform=dataset_transforms,download=True)
test_set = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, transform=dataset_transforms,download=True)

# img,target = train_set[0]
# print(test_set[0])
# print(test_set.classes)
# img.show()
writer = SummaryWriter("p11")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()