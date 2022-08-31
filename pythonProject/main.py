# 导入相关模块
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import torch.optim as optim
import torch.nn as nn
from network import Net
from cifar_network import CifarNet
import torchvision
import matplotlib.pyplot as plt


class ParamData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, inter):
        self.root_dir = root_dir
        self.paramdata = os.listdir(self.root_dir)[:inter]
        self.labeldata = os.listdir(self.root_dir)[inter:]

    def __len__(self):
        return len(self.paramdata)

    def __getitem__(self, index):
        data_index = self.paramdata[index]
        label_index = self.labeldata[index]
        data_path = os.path.join(self.root_dir, data_index)
        label_path = os.path.join(self.root_dir, label_index)
        par = torch.load(data_path).detach()
        lab = torch.load(label_path).detach()
        return par, lab


class ParamData_test(Dataset):  # 继承Dataset
    def __init__(self, root_dir, inter1, inter2, inter3):
        self.root_dir = root_dir
        self.paramdata = os.listdir(self.root_dir)[:inter1]
        self.labeldata = os.listdir(self.root_dir)[inter1:inter2]
        self.fcpara = os.listdir(self.root_dir)[inter2:inter3]
        self.fcbias = os.listdir(self.root_dir)[inter3:]

    def __len__(self):
        return len(self.paramdata)

    def __getitem__(self, index):
        data_index = self.paramdata[index]
        label_index = self.labeldata[index]
        fcpara_index = self.fcpara[index]
        fcbias_index = self.fcbias[index]
        data_path = os.path.join(self.root_dir, data_index)
        label_path = os.path.join(self.root_dir, label_index)
        fcpara_path = os.path.join(self.root_dir, fcpara_index)
        fcbias_path = os.path.join(self.root_dir, fcbias_index)
        par = torch.load(data_path).detach()
        lab = torch.load(label_path).detach()
        fcpara = torch.load(fcpara_path).detach()
        fcbias = torch.load(fcbias_path).detach()

        return par, lab, fcpara, fcbias


if __name__ == '__main__':
    device = torch.device('cuda:0')
    net = Net()
    net.to(device)
    train = ParamData(r'D:\pythonProject\traindataset', 9)
    trainloader = DataLoader(train, batch_size=1, shuffle=False)
    # test = ParamData_test(r'D:\pythonProject\testdataset', 2, 4, 6)
    test = ParamData_test(r'D:\pythonProject\testdataset111', 1, 2, 3)
    testloader = DataLoader(test, batch_size=1, shuffle=False)
    # for i_batch, batch_data in enumerate(dataloader):
    #     print(i_batch)
    #     print(torch.mean(batch_data[0]))
    #     print(torch.mean(batch_data[1]))

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
    for epoch in range(50):
        for i, data in enumerate(trainloader, 0):
            inputs, gth = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, gth)
            loss.backward()
            optimizer.step()
            # print("epoch:", epoch, loss.item())

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, gth, fcpara, fcbias = data
                # inputs, gth = data
                outputs = net(inputs)
                loss = criterion(outputs, gth)
                print("test loss:", loss.item())

                param3 = outputs.view(3, 3, 5, 5)
                param1 = inputs[:, :, :225]
                param1 = param1.reshape(3, 3, 5, 5)
                param2 = inputs[:, :, 225:]
                param2 = param2.reshape(3, 3, 5, 5)
                gth_para = gth.reshape(3, 3, 5, 5)
                cifarnet = CifarNet()
                for n, p in cifarnet.named_parameters():
                    # print(n, p.size())
                    if n == 'conv1.weight':
                        p.data = param1
                    elif n == 'conv2.weight':
                        p.data = param2
                    elif n == 'conv3.weight':
                        # p.data = gth_para
                        p.data = param3
                    elif n == 'fc1.weight':
                        p.data = fcpara.squeeze(0)
                    elif n == 'fc1.bias':
                        p.data = fcbias

                cifarnet.to(device)
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                testset_cifar = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                             download=False, transform=transform)
                testloader_cifar = torch.utils.data.DataLoader(testset_cifar, batch_size=4,
                                                               shuffle=False, num_workers=0)
                classes = ('plane', 'car', 'bird', 'cat',
                           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader_cifar:
                        images_cifar, labels_cifar = data
                        images_cifar, labels_cifar = images_cifar.cuda(), labels_cifar.cuda()
                        outputs_cifar = cifarnet(images_cifar)
                        _, predicted_cifar = torch.max(outputs_cifar.data, 1)
                        predicted_cifar = predicted_cifar.to(device)
                        total += labels_cifar.size(0)
                        correct += (predicted_cifar == labels_cifar).sum().item()
                print('accuracy: %d %%' % (100 * correct / total))

    print('finish training')










    

