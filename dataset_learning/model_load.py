import torch
import torchvision
from torch import nn
from torch.nn import Flatten

# 加载保存方式1的
model1 = torch.load("vgg16_method1.pth")
print(model1)

# 加载保存方式2的
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

#陷阱

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

model3 = torch.load("yuyu1.pth")
print(model3)