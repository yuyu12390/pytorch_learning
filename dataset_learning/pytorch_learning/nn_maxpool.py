import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset=test_set,batch_size=64)



class Yuyu(nn.Module):
    def __init__(self):
        super(Yuyu,self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return output

yuyu = Yuyu()

writer = SummaryWriter("../maxpool_log")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images('input',imgs,step)
    output = yuyu(imgs)
    print(output.shape)
    writer.add_images('output', output, step)
    step += 1

writer.close()

