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
from sklearn.model_selection import train_test_split
class DatasetMnist(Dataset):
    def __init__(self, root, target=None, train=True, split_train=0):
        self.root = root
        self.labels = None
        if train:
            path = os.path.join(root, r"D:\basic_Python\basic_Python\deep learning basic\RESNET\ece597697-sp2023\train.csv\train.csv")
            df = pd.read_csv(path)
            sz = len(df)
            cach = int(sz * 0.9)
            if not split_train:
                self.id = df['id']
                df = df.drop(columns = ['id'])
                self.images = np.array(df.drop(columns=[target]))
                self.labels = np.array(df[target])
                self.labels = torch.from_numpy(self.labels)
                self.images = torch.from_numpy(self.images)
            elif split_train == 1:
                self.id = df['id'][:cach]
                df = df.drop(columns = ['id'])
                self.images = np.array(df.drop(columns=[target])[:cach])
                self.labels = np.array(df[target].iloc[:cach])
                self.labels = torch.from_numpy(self.labels)
                self.images = torch.from_numpy(self.images)
            else:
                self.id = df['id'][cach:]
                df = df.drop(columns = ['id'])
                self.images = np.array(df.drop(columns=[target])[cach:])
                self.labels = np.array(df[target].iloc[cach:])
                self.labels = torch.from_numpy(self.labels)
                self.images = torch.from_numpy(self.images)
                
        else: 
            path = os.path.join(root, r"D:\basic_Python\basic_Python\deep learning basic\RESNET\ece597697-sp2023\test.csv\test.csv")
            df = pd.read_csv(path)
            self.id = df['id']
            self.images = np.array(df.drop(columns=['id']))
            self.images = torch.from_numpy(self.images)
        # print(self.images[0])
        self.images = self.images / 255
        self.images =  self.images.reshape(-1, 1, 28, 28)
        # print(self.images[0])
        # self.images = F.resize(self.images, size=(224, 224), interpolation=F.InterpolationMode.BILINEAR)
        self.len = len(self.images)
    def __len__(self):
        return self.len
    def get_id(self):
        return self.id
    def __getitem__(self, idx):
        if self.labels != None:
            label = self.labels[idx]
        image = self.images[idx]
        # print(image)
        
        image = F.resize(image, size=(224, 224), interpolation=F.InterpolationMode.BILINEAR)
        if self.labels != None:
            return image, label
        else: return image
