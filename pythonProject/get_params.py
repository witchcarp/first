import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cifar_network import CifarNet

device = torch.device('cuda:0')
model = CifarNet()
model.load_state_dict(torch.load('cifar10_params.pth', map_location='cuda:0'))
model.to(device)
params = list([])
# for n, p in model.named_parameters():
#     print(n, p.size())

for n, p in model.named_parameters():
    if n in ['conv1.weight', 'conv2.weight', 'conv3.weight']:
        p = p.view(1, -1)
        params.append(p)
    elif n == 'fc1.weight':
        torch.save(p, 'testdataset/c1.pt')
    elif n == 'fc1.bias':
        torch.save(p, 'testdataset/d1.pt')
params1 = torch.tensor(params[0])
params2 = torch.tensor(params[1])
params3 = torch.tensor(params[2])

params = torch.cat((params1, params2), dim=1)
torch.save(params, 'testdataset/a1.pt') # params of 1,2 layers
torch.save(params3, 'testdataset/b1.pt') # params of 3 layers













