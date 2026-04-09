import torch
import torchvision
from torch import nn
from torch.nn import Flatten
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data_cifar10",train=True,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=1)

class Yuyu(nn.Module):
    def __init__(self):
        super(Yuyu,self).__init__()
        self.conv1 = nn.Conv2d(3,32,5,padding = 2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,32,5,padding = 2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32,64,5,padding = 2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024,64)
        self.linear2 = nn.Linear(64,10)

        self.model1 = nn.Sequential(
            nn.Conv2d(3,32,5,padding = 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding = 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding = 2),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output

loss = nn.CrossEntropyLoss()
yuyu = Yuyu()
for data in dataloader:
    imgs,targets = data
    outputs = yuyu(imgs)
    result_loss = loss(outputs,targets)
    # result_loss.backward()
    print("OK")
    # print(result_loss)


