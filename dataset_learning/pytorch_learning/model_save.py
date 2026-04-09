import torch
import torchvision
from torch import nn
from torch.nn import Flatten

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1（模型结构+参数）
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2(模型参数)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

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

yuyu = Yuyu()
torch.save(yuyu, "yuyu1.pth")