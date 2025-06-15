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
from CNN import *
from Dataset import DatasetMnist
from argparse import ArgumentParser
import shutil
import os
from resnet import Resnet
from sklearn.metrics import accuracy_score

def get_args():
    parser = ArgumentParser(description="CNN trainning")
    parser.add_argument("--root", "-r", type=str, default="", help="Root of the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="The number of epoch")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="The size of batch")
    parser.add_argument("--image_size", "-i", type=int, default=224, help="The size of image")
    parser.add_argument("--logging", "-l", type=str, default="Tensorboard")
    parser.add_argument("--checkpoint", "-c", type=str, default="SaveModel/best_cnn.pt")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_args()
    data_test = DatasetMnist(root=args.root, train=False)
    test_loader = DataLoader(dataset=data_test, batch_size=args.batch_size, drop_last=False, pin_memory=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = Resnet().to(device)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("ko co best model")
        exit
    
    all_predictions = pd.DataFrame(columns=['id', 'label'])

    for images in test_loader:
        images = images.to(device)
        output = model(images)
        with torch.no_grad():
            indices = torch.argmax(output.cpu(), axis=1)
            indices = indices.numpy()
            indices = pd.DataFrame(indices, columns=['label'])
            all_predictions = pd.concat([all_predictions, indices], axis=0, ignore_index=True)
    print(all_predictions)
    all_predictions['id'] = data_test.get_id().reset_index(drop=True)
    print(data_test.get_id())
    all_predictions.to_csv('submission2.csv', index=False)
    print(all_predictions['id'])
