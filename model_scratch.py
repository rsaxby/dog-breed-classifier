#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import math
# define the CNN architecture
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
        ## Define layers of a CNN
        # conv layers
        self.conv3_32 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv32_32 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv32_64 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv64_64 = nn.Conv2d(64, 64, 3, padding=1)
        # pooling layer
        self.pool = nn.MaxPool2d(2,2)
        # dropout
        self.dropout = nn.Dropout(0.5)
        # fc layers
        self.fc1 = nn.Linear(64*28*28, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        # initialize weights
        # for every module
        # m.weight.data shoud be taken from a normal distribution
        # m.bias.data should be 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # get the number of the inputs
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                y = math.sqrt(2./n)
                m.weight.data.normal_(0, y )
                m.bias.data.fill_(0)
                
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv3_32(x))
        x = self.pool(x)
        x = F.relu(self.conv32_32(x))
        x = self.pool(x)
        x = F.relu(self.conv32_64(x))
        x = self.pool(x)
        # flatten for fc layers
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x