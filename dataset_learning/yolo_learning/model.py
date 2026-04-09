import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from yolo_learning.yolo_datasets import YOLODataset


class yuyuModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return  F.relu(self.conv2(x))

if __name__ == "__main__":
    model = yuyuModule()
    dataset = YOLODataset(r"D:\yolo_datasets\HelmetDataset-YOLO-Train\images",r"D:\yolo_datasets\HelmetDataset-YOLO-Train\labels",
                          transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                    torchvision.transforms.Resize((512,512))]),label_transform=None)
    image,target = dataset[0]
    image = image.unsqueeze(0)
    output = model(image)
    # print(output)
    # print(model)
    torch.onnx.export(model,image,"yuyu.onnx")