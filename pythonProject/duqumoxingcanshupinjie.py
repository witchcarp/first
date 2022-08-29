import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cifar_network import Net

device = torch.device('cuda:0')
model = Net()
model.load_state_dict(torch.load('cifar2_params.pth', map_location='cuda:0'))
model.to(device)
params = list([])
# for n, p in model.named_parameters():
#     print(n, p)

for n, p in model.named_parameters():
    if n in ['conv1.weight', 'conv2.weight', 'conv3.weight']:
        p = p.view(1, -1)
        params.append(p)
params1 = torch.tensor(params[0])
params2 = torch.tensor(params[1])
params3 = torch.tensor(params[2])

params = torch.cat((params1, params2), dim=1)
torch.save(params, 'dataset/a2.pt') # params of 1,2 layers
torch.save(params3, 'dataset/b2.pt') # params of 3 layers













