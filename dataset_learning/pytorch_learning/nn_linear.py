import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64,drop_last=True)

class Yuyu(nn.Module):
    def __init__(self):
        super(Yuyu,self).__init__()
        self.linear1 = nn.Linear(196608,10)

    def forward(self, input):
        output = self.linear1(input)
        return output

yuyu = Yuyu()

for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = yuyu(output)
    print(output.shape)