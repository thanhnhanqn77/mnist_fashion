import pandas as pd
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import os
from torch.nn.functional import cross_entropy
from torch.optim import SGD
import torchvision.datasets.stanford_cars as cars_data
import torchvision.transforms.functional as F
import PIL
from torchvision import transforms
from tqdm.autonotebook import tqdm 
from torch.utils.tensorboard import SummaryWriter
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
        self.downsample = downsample
    def forward(self, x):
        identity = x
        x = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x+=identity
        x = self.relu(x)
        return x
class Resnet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block1 = self.MakeLayer(3, 64, 64)
        
        self.block2 = self.MakeLayer(4, 64, 128, 2)

        self.block3 = self.MakeLayer(6, 128, 256, 2)
            

        self.block4 = self.MakeLayer(3, 256, 512, 2)

        self.block5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )
    def MakeLayer(self, times, in_channels, out_channels, stride=1, padding=1):
        lis = []
        downsample = None
        if in_channels != out_channels or stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels)
            )    
        lis.append(BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample))
        in_channels = out_channels        
        for i in range(1, times):
            lis.append(BasicBlock(in_channels, out_channels))
        return nn.Sequential(*lis)
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
