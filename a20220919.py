
abcdefg
xueAAAAAAAAA
xxxxxxx更新




lia对xue的修改》》》》》》



import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


