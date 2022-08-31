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


class ParamData(Dataset):  # 继承Dataset
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.paramdata = os.listdir(self.root_dir)[:9]
        self.labeldata = os.listdir(self.root_dir)[9:]

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


if __name__ == '__main__':
    device = torch.device('cuda:0')
    net = Net()
    net.to(device)
    data = ParamData(r'/traindataset')
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    # for i_batch, batch_data in enumerate(dataloader):
    #     print(i_batch)
    #     print(torch.mean(batch_data[0]))
    #     print(torch.mean(batch_data[1]))

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            inputs, gth = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, gth)
            loss.backward()
            optimizer.step()
            print(epoch, loss)

    print('finish training')


