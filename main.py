# imports

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

"""
Here every type on argument that the user can send in is declared
"""

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

"""
Here we define the amount of cell classes and the files to be used in the training and testing dataset.
We also define the amount of number k-fold validation.
"""
num_classes = 4
class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']
shuffle = True
k = 5 # Cross-validation splits
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

"""
Class: AddGausianNoise
Function: Adds Gaussian noise to reduce overfitting
Input to constructor: mean and standard deviation
Input to call: tensor we want to add gausian noise to
Output of call: tensor with added gaussian noise
"""
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


"""
Name: Main
Function: Test 
Input: Test
Output: Test
"""

def main():
    # gets arguments
    args = parser.parse_args()
    # prints cuda version and device
    print(torch.version.cuda)
    print(torch.cuda.FloatTensor([1.]))

    # prints image size and type of filter
    image_size = args.image_size
    print('Using image size: ', image_size)
    print('Using filter: ', args.filter)

    # Define the transforms to be applied to the train and validation set
    # Training set has data augmentation transforms while validation set does not
    train_tsfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size+math.floor(0.1*image_size), interpolation=PIL.Image.BICUBIC), # increases size by 1%, new pixels are interpolated (estimated) bicubicly
        transforms.RandomResizedCrop(image_size), # takes a random crop from the new larger image
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.1) # adds Gaussian noise
        #normalize,
    ])
    val_test_tsfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        #normalize,
    ])

    # Load training images and corresponding labels
    train_images, train_labels = parseData(basePath=args.data,filter_name=args.filter, label_paths=train_label_paths, class_names=class_names,set_name="Training set")

    # Remove all images in training set with labels 4 ('apoptosis / civiatte body')
    for i in range(len(train_labels)-1, -1, -1):
        if(train_labels[i] == 4):
            train_labels.pop(i)
            train_images.pop(i)

    # Load testing images and corresponding labels
    test_images, test_labels = parseData(basePath=args.data, filter_name=args.filter, label_paths=test_label_paths, class_names=class_names,set_name="Testing set")

    # Remove all images in training set with labels 4 ('apoptosis / civiatte body')
    for i in range(len(test_labels)-1, -1, -1):
        if(test_labels[i] == 4):
            test_labels.pop(i)
            test_images.pop(i)

    # Upsamples the training data if args.upsample = True
    # adds all images in class 0, 8 times in total it appears in the dataset 9 times
    # adds all images in class 1, 4 times in total it appears in the dataset 5 times
    # adds all images in class 2, 1 time in total it appears in the dataset 2 times
    # adds all images in class 3, 1 time, but only the 2000 first images
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

        print(class_names[0], "appears", len(c0_ind), "times in the non augmented training set")
        print(class_names[0], "appears", len([i for i, x in enumerate(train_labels) if x == 0]), "times in the augmented training set")
        print(class_names[1], "appears", len(c1_ind), "times in the non augmented training set")
        print(class_names[1], "appears", len([i for i, x in enumerate(train_labels) if x == 1]), "times in the augmented training set")
        print(class_names[2], "appears", len(c2_ind), "times in the non augmented training set")
        print(class_names[2], "appears", len([i for i, x in enumerate(train_labels) if x == 2]), "times in the augmented training set")
        print(class_names[3], "appears", len(c3_ind), "times in the non augmented training set")
        print(class_names[3], "appears", len([i for i, x in enumerate(train_labels) if x == 3]), "times in the augmented training set")

    # Shuffles the training set
    temp = list(zip(train_labels, train_images))
    random.shuffle(temp)
    train_labels, train_images = zip(*temp)

    # Creates a StratifiedKfold Object with the number of splits, shuffle option and seed
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=args.seed)

    # Define device for the tensors
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn')

    # Transform the test dataset to torch tensor, x is images and y is labels
    tensor_test_images = torch.tensor(test_images, dtype=torch.float32, device=device)
    tensor_test_labels = torch.tensor(test_labels, dtype=torch.long, device=device)
    tensor_test_images = tensor_test_images.permute(0, 3, 1, 2)

    split_counter = 0
    # Generates indices and data split into training and test set
    for train, val in skf.split(train_images, train_labels):
        # Transform the training and validation dataset to torch tensors, according to current split
        # x is images and y is labels
        tensor_train_images = torch.tensor([train_images[i] for i in train], dtype=torch.float32, device=device)
        tensor_val_images = torch.tensor([train_images[i] for i in val], dtype=torch.float32, device=device)
        tensor_train_labels = torch.tensor([train_labels[i] for i in train], dtype=torch.long, device=device)
        tensor_val_labels = torch.tensor([train_labels[i] for i in val], dtype=torch.long, device=device)

        # Order array dimensions to pytorch standard
        # Changing format from <batch size, image height, image width, image channel>
        # to <batch size, image channel, image height, image width>.
        tensor_train_images = tensor_train_images.permute(0, 3, 1, 2)
        tensor_val_images = tensor_val_images.permute(0, 3, 1, 2)

        # Creates 3 CellDataset Objects with the corresponding transform and image- and label tensors, to be able to use DataLoader
        train_dataset = CellDataset(tensors=(tensor_train_images, tensor_train_labels),
                                    transform=train_tsfm)
        val_dataset = CellDataset(tensors=(tensor_val_images, tensor_val_labels),
                                  transform=val_test_tsfm)
        test_dataset = CellDataset(tensors=(tensor_test_images, tensor_test_labels),
                                   transform=val_test_tsfm)

        # Prints the sizes of the three datasets
        train_dataset_size = len(train_dataset)
        val_dataset_size = len(val_dataset)
        test_dataset_size = len(test_dataset)
        print("train size: {}".format(train_dataset_size))
        print("val size: {}".format(val_dataset_size))
        print("test size: {}".format(test_dataset_size))

        # Creates DataLoaders for each dataset to use for training
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
        #Creates a dictionary with the three dataloaders
        loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
        #Sarts training for each k-fold and saves it in model
        model = run_model(loaders, split_counter, args, class_names)
        split_counter += 1

    # View results of model
    # visualize_model(model, my_dataloader)
    # plt.show()

    # View single image
    # crop = Image.fromarray(images[5814])
    # crop.show()
    # print(labels[5814])

if __name__ == '__main__':
    main()
