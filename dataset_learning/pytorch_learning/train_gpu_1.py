import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

#准备数据集
train_data = torchvision.datasets.CIFAR10("./data_cifar10",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data =  torchvision.datasets.CIFAR10("./data_cifar10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("训练测试集的长度为:{}".format(test_data_size))

#利用dataloader加载数据集
train_loader = DataLoader(dataset=train_data,batch_size=64)
test_loader = DataLoader(dataset=test_data,batch_size=64)

#创建网络模型
class Yuyu(nn.Module):
    def __init__(self):
        super(Yuyu,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.model(x)
        return x

yuyu = Yuyu()
if torch.cuda.is_available():
    yuyu = yuyu.cuda()
#损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(yuyu.parameters(),lr=learning_rate)

#设置训练网络的参数
#记录训练次数
total_train_step = 0
#记录测试次数
total_test_step = 0
#训练轮数
epoch = 10

#添加tensorboard
writer = SummaryWriter("../model_log")
start_time = time.time()

for i in range(epoch):
    print("-------------第{}轮训练开始-------------".format(i+1))
    #训练开始
    for data in train_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = yuyu(imgs)
        loss = loss_fn(outputs, targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数：{},loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    #测试开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = yuyu(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = ((outputs.argmax(dim=1) == targets).sum())
            total_accuracy = accuracy + total_accuracy
    total_test_step += 1
    print("整体测试集的loss:{}".format(total_test_loss))
    print("整体测试集的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    torch.save(yuyu,"yuyu_{}.pth".format(i))
    print("模型已保存")


writer.close()