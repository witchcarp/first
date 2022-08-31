

import torch.nn as nn
import torch.nn.functional as F


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5, bias=False)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 3, 5, bias=False)
        self.conv3 = nn.Conv2d(3, 3, 5, bias=False)
        self.fc1 = nn.Linear(20 * 20 * 3, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 20 * 20 * 3)
        x = self.fc1(x)


        return x



