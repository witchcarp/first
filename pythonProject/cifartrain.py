import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from cifar_network import CifarNet

# 在做数据归一化之前必须要把PIL Image转成Tensor，而其他resize或crop操作则不需要。
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np




net = CifarNet()

# 定义损失函数和优化器

import torch.optim as optim

# Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练网络
for epoch in range(4):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()  # 防止梯度累积
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Trainning')

# torch.save(net.state_dict(), 'cifar11_params.pth')


# 在GPU上训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



net.to(device)
# 将输入放到GPU


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        for n, p in net.named_parameters():
            if n == 'conv3.weight':
                p.zero_()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.to(device)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('accuracy: %d %%' % (100 * correct / total))
