import os
import matplotlib.image as mpimg
import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import torchvision
from torch.utils import data
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import time
import random

from parseData import parseData
from efficientnet_pytorch import EfficientNet

from visualize_model import visualize_model
from train_valid_split import train_valid_split
from run_model import run_model
from CellDataset import CellDataset

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch EfficientNet Training')
parser.add_argument('--data', metavar='DIR', default="KI-dataset-4-types/All_Slices/",
                    help='path to KI-Dataset folder')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    help='model architecture (default: efficientnet-b0)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default:8), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('-val', '--validate', dest='validate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--feature_extract', dest='feature_extract',
                    action='store_true',
                    help="Train only last layer (otherwise full model)")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=32, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--upsample', default=True, action='store_true',
                    help='upsample, else use class weights')
parser.add_argument('--filter', default="",
                    help='filter we want to use for training the model')
parser.add_argument('--outdest', default="",
                    help='where we want to save our output data')

# Static config
num_classes = 4
class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']

train_label_paths = [
    "P19_1_1",
    "P19_1_2",
    "P19_2_1",
    "P19_2_2",
    "P19_3_1",
    "P19_3_2",
    "P20_1_3",
    "P20_1_4",
    "P20_2_2",
    "P20_2_3",
    "P20_2_4",
    "P20_3_1",
    "P20_3_2",
    "P20_3_3",
    "P20_4_1",
    "P20_4_2",
    "P20_4_3",
    "P20_5_1",
    "P20_5_2",
    "P20_6_1",
    "P20_6_2",
    "P20_7_1",
    "P20_7_2",
    "P20_8_1",
    "P20_8_2",
    "P20_9_1",
    "P20_9_2",
    "P9_1_1",
    "P9_2_1",
    "P9_2_2",
    "P9_3_1",
    "P9_3_2",
    "P9_4_1",
    "P9_4_2"
]

test_label_paths = [
    "N10_1_1",
    "N10_1_2",
    "N10_1_3",
    "N10_2_1",
    "N10_2_2",
    "P13_1_1",
    "P13_1_2",
    "P13_2_1",
    "P13_2_2",
    "P28_7_5",
    "P28_8_5",
    "P28_10_4",
    "P28_10_5",
]

shuffle = True
k = 5 # Cross-validation splits

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def lambdaTransform(image):
    return image * 2.0 - 1.0

def main():
    args = parser.parse_args()
    print(torch.version.cuda)
    a = torch.cuda.FloatTensor([1.])
    print(a)

    image_size = args.image_size
    print('Using image size: ', image_size)
    filter = args.filter
    print('Using filter ', filter)

    train_tsfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size+math.floor(0.1*image_size), interpolation=PIL.Image.BICUBIC),
        transforms.RandomResizedCrop(image_size),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.1)
        #normalize,
    ])

    val_tsfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        #normalize,
    ])

    # Load and split datasets and convert to tensor
    # Test images from different slices than train

    train_images, train_labels = parseData(basePath=args.data,filter_name=filter, label_paths=train_label_paths, class_names=class_names,et_name="Training set")

    # remove all images in training set with labels 4
    for i in range(len(train_labels)-1, -1, -1):
        if(train_labels[i] == 4):
            train_labels.pop(i)
            train_images.pop(i)

    test_images, test_labels = parseData(basePath=args.data, filter_name=filter, label_paths=test_label_paths, class_names=class_names,et_name="Testing set")

    for i in range(len(test_labels)-1, -1, -1):
        if(test_labels[i] == 4):
            test_labels.pop(i)
            test_images.pop(i)

    # Upsamples the training data if args.upsample = True
    if args.upsample:
        c0_ind = [i for i, x in enumerate(train_labels) if x == 0]
        c1_ind = [i for i, x in enumerate(train_labels) if x == 1]
        c2_ind = [i for i, x in enumerate(train_labels) if x == 2]
        c3_ind = [i for i, x in enumerate(train_labels) if x == 3]

        for i in range(8):
            for idx, val in enumerate(c0_ind):
                train_labels.append(train_labels[val])
                train_images.append(train_images[val])
        for i in range(4):
            for idx, val in enumerate(c1_ind):
                train_labels.append(train_labels[val])
                train_images.append(train_images[val])
        for i in range(1):
            for idx, val in enumerate(c2_ind):
                train_labels.append(train_labels[val])
                train_images.append(train_images[val])

        for idx, val in enumerate(c3_ind):
            if idx < 2000:
                train_labels.append(train_labels[val])
                train_images.append(train_images[val])

    temp = list(zip(train_labels, train_images))
    random.shuffle(temp)
    train_labels, train_images = zip(*temp)

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn')

    # Transform to torch tensor
    tensor_test_x = torch.tensor(test_images, dtype=torch.float32, device=device)
    tensor_test_y = torch.tensor(test_labels, dtype=torch.long, device=device)
    tensor_test_x = tensor_test_x.permute(0, 3, 1, 2)

    split = 0

    for train, val in skf.split(train_images, train_labels):
        tensor_train_x = torch.tensor([train_images[i] for i in train], dtype=torch.float32, device=device)
        tensor_val_x = torch.tensor([train_images[i] for i in val], dtype=torch.float32, device=device)
        tensor_train_y = torch.tensor([train_labels[i] for i in train], dtype=torch.long, device=device)
        tensor_val_y = torch.tensor([train_labels[i] for i in val], dtype=torch.long, device=device)

        # Order array dimensions to pytorch standard
        tensor_train_x = tensor_train_x.permute(0, 3, 1, 2)
        tensor_val_x = tensor_val_x.permute(0, 3, 1, 2)


        train_dataset = CellDataset(tensors=(tensor_train_x, tensor_train_y),
                                    transform=train_tsfm)
        val_dataset = CellDataset(tensors=(tensor_val_x, tensor_val_y),
                                  transform=val_tsfm)
        test_dataset = CellDataset(tensors=(tensor_test_x, tensor_test_y),
                                   transform=val_tsfm)

        # Sizes of datasets
        train_dataset_size = len(train_dataset)
        val_dataset_size = len(val_dataset)
        test_dataset_size = len(test_dataset)
        print("train size: {}".format(train_dataset_size))
        print("val size: {}".format(val_dataset_size))
        print("test size: {}".format(test_dataset_size))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.workers, pin_memory=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

        loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }

        model = run_model(loaders, split, args, class_names)
        split += 1

    # View results of model
    # visualize_model(model, my_dataloader)
    # plt.show()

    # View single image
    # crop = Image.fromarray(images[5814])
    # crop.show()
    # print(labels[5814])

if __name__ == '__main__':
    main()
