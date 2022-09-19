import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.fclayer1 = nn.Linear(4608, 4096)
        self.fclayer2 = nn.Linear(4096, 2048)
        self.fclayer3 = nn.Linear(2048, 1728)
        self.norm1 = nn.LayerNorm([3])
        self.dropout = nn.Dropout(0.7)

    def forward(self, input):
        num_chan = input.shape[1]
        num_size = input.shape[0]
        idx = self.in_channels_list.index(num_chan)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation)
        y = y.squeeze(3).reshape(1, -1)

        y = self.fclayer1(F.relu(y))
        y = self.dropout(self.fclayer2(F.relu(y)))
        y = self.fclayer3(F.relu(y))

        return y



class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list

    def forward(self, input):
        num_chan = input.shape[0]
        idx = self.in_features_list.idx(num_chan)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


