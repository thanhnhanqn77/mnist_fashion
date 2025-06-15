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
from sklearn.metrics import accuracy_score

def get_args():
    parser = ArgumentParser(description="CNN trainning")
    parser.add_argument("--root", "-r", type=str, default="", help="Root of the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="The number of epoch")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="The size of batch")
    parser.add_argument("--image_size", "-i", type=int, default=224, help="The size of image")
    parser.add_argument("--logging", "-l", type=str, default="Tensorboard")
    parser.add_argument("--checkpoint", "-c", type=str, default="SaveModel/cur_cnn.pt")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    data_train = DatasetMnist(root=args.root, target='label', train=True, split_train=1)
    data_eval = DatasetMnist(root=args.root, target='label', train=True, split_train=2)

    data_test = DatasetMnist(root=args.root, train=False)
    train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    eval_loader = DataLoader(dataset=data_eval, batch_size=args.batch_size, drop_last=False, pin_memory=True)
    # image, label = data_train.__getitem__(18)
    # image_pil = transforms.ToPILImage()(image)
    # image_pil.show()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    best_accuracy = 0

    
    if os.path.isdir("Tensorboard"):
        shutil.rmtree("Tensorboard")
    if not os.path.isdir("SaveModel"):
        os.mkdir("SaveModel")
    
    model = Resnet().to(device)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_iters = len(train_loader)
    writer = SummaryWriter("Tensorboard")
    # print(args.checkpoint)
    # checkpoint = torch.load(args.checkpoint)
    # print(checkpoint["epoch"])
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        # best_loss = checkpoint["best_loss"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch=0
    print(start_epoch)
        
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, colour='green')
        for iter, (images, labels) in enumerate (progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)        

            loss_value = cross_entropy(output, labels)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            if iter % 10 == 0:
                progress_bar.set_description(f"Epoch {epoch + 1}/{args.epochs}. Iter {iter}. Loss {loss_value}")
            writer.add_scalar('Train/Loss', loss_value, iter + (epoch) * num_iters)
        model.eval()
        all_predictions = []
        all_labels = []

        for iter, (images, labels) in enumerate(eval_loader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions.cpu(), dim = 1)
                all_predictions.extend(indices)
        checkpoint  = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, r"SaveModel/cur_cnn.pt")
        
        all_labels = [label.item() for label in all_labels]
        all_predictions = [predict.item() for predict in all_predictions]
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"epoch {epoch + 1}. Accuracy {accuracy}")
        writer.add_scalar("Val/Accuracy", accuracy, epoch)
        

        if accuracy < best_accuracy:
            checkpoint  = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_accuracy": accuracy
            }
            torch.save(checkpoint, r"SaveModel/best_cnn.pt")
            best_accuracy = accuracy
        
    # writer.close()


