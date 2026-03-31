import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import dataloader, DataLoader
from torch.utils.tensorboard import SummaryWriter

#测试数据集
dataset = torchvision.datasets.CIFAR10(root='./data_cifar10',train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset=dataset,batch_size=64)

class Yuyu(nn.Module):
    def __init__(self):
        super(Yuyu,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x
yuyu = Yuyu()
print(yuyu)

writer = SummaryWriter("conv")

step = 0
for data in dataloader:
    imgs,targets = data
    output = yuyu(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images('input',imgs,step)

    output = torch.reshape(output,(-1,3,30,30))
    print(output.shape)
    writer.add_images('output', output, step)
    step = step + 1

