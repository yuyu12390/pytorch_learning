import torch
import torchvision
from tensorboard import summary
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nn_conv2d import targets

input = torch.tensor([[1,-0.5],
                     [-1,3]])

output = torch.reshape(input,(-1,1,2,2))


dataset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64)

print(output.shape)

class Yuyu(nn.Module):
    def __init__(self):
        super(Yuyu,self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

yuyu = Yuyu()#创建神经网络

writer = SummaryWriter("relu_log")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = yuyu(imgs)
    writer.add_images("output",output,step)
    step+=1

writer.close()

