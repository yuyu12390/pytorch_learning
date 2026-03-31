import torch
from torch import nn

class Yuyu(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

yuyu = Yuyu() #创建神经网络
x = torch.tensor(1.0)
output = yuyu(x)
print(output)