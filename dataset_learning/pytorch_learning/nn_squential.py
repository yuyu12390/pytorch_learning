import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten
from torch.utils.tensorboard import SummaryWriter


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
        output = self.conv1(input)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.maxpool3(output)
        output = self.flatten(output)
        output = self.linear1(output)
        output = self.linear2(output)
        return output

yuyu = Yuyu()

print(yuyu)
input = torch.ones(64,3,32,32)
output = yuyu(input)
print(output.shape)

writer = SummaryWriter("../sequ_log")
writer.add_graph(yuyu,input)

writer.close()