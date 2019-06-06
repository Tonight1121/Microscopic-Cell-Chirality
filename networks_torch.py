import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import cv2
import sys

import os
import tensorflow as tf
import time
import copy


import numpy as np


###################################################################


class MIC_3D(nn.Module):

    def __init__(self, num_classes=3):
        super(MIC_3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(16)
        self.bn3 = nn.BatchNorm3d(8)
        self.bn4 = nn.BatchNorm3d(4)
        self.bn5 = nn.BatchNorm3d(1)

        self.conv3d_1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 7, 7), stride=(1, 1, 1), padding=(1, 3, 3))
        self.conv3d_2 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.conv3d_3 = nn.Conv3d(in_channels=16, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.conv3d_4 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.conv3d_5 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)

        self.conv3d_6 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.maxpool = nn.MaxPool3d((3, 3, 3), stride=(1, 1, 1), padding=1)

        self.drop2d_1 = nn.Dropout2d(0.1)

        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('x 3d1 shape {}'.format(x.shape))

        x = self.conv3d_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print('x 3d2 shape {}'.format(x.shape))

        x = self.conv3d_3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # print('x 3d3 shape {}'.format(x.shape))

        x = self.conv3d_4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # print('x 3d4 shape {}'.format(x.shape))

        # x = self.conv3d_5(x)
        # x = self.bn5(x)
        # x = self.relu(x)
        # print('x 3d5 shape {}'.format(x.shape))

        x = self.avgpool(x)
        # print('x avgpool shape {}'.format(x.shape))

        # x = self.conv3d_6(x)
        # x = self.relu(x)
        # x = self.bn4(x)
        # print('x 3d6 shape {}'.format(x.shape))

        x = x.view(x.size(0), -1)
        # print('x flatten shape {}'.format(x.shape))

        # time.sleep(30)
        # x = self.fc1(x)
        # x = self.relu(x)
        # print('x fc1 shape {}'.format(x.shape))

        x = self.fc2(x)
        x = self.relu(x)
        # print('x fc2 shape {}'.format(x.shape))
        # print('network output shape {}'.format(x.shape))
        # time.sleep(30)
        return x

