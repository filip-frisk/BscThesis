# A varitation of the emsemble.py file, used for esembling 4 models at a time
# (this was the ensemble used in the final report)
# Imports
import torch
from run_model import AverageMeter, ProgressMeter, confusion_matix_string,accuracy
from sklearn.metrics import classification_report
from CustomDataset import CustomDataset
from parseData import parseData
import argparse
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, models, transforms
import torch.multiprocessing as mp
import PIL
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import numpy as np


#Parser arguments
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
parser.add_argument('--model_paths', default=[],
                    help='path to models to be used')

# Variables for experiment
num_classes = 4
class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']
filters = ["","_Macenko","_Reinhard","_SCD"]
y_pred_per_filter = {"True" : []}
p_pred_per_filter = {}
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

def main():
    k = 4 # picks the k-th best model from training

    args = parser.parse_args()
    model_paths = [item for item in args.model_paths.split(",")]
    print(model_paths)

    # Define device for the tensors and enables multiprocessing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn')

    for filter, path in zip(filters, model_paths):

        # prints the current filter to the terminal
        if filter: print(filter.strip('_'))
        else: print("No_filter")

        #Hardcoded destination from this project repository
        #filename = "model_best_{}.pth.tar".format(k)
        filename = "model_best.pth.tar"

        test_images, test_labels = parseData(basePath=args.data, filter_name=filter, label_paths=test_label_paths,
                                             class_names=class_names, set_name="Testing set")

        # Transform the test dataset to torch tensor, x is images and y is labels
        tensor_test_images = torch.tensor(test_images, dtype=torch.float32, device=device)
        tensor_test_labels = torch.tensor(test_labels, dtype=torch.long, device=device)
        tensor_test_images = tensor_test_images.permute(0, 3, 1, 2)

        #Define transform
        test_tsfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])

        test_dataset = CustomDataset(tensors=(tensor_test_images, tensor_test_labels),
                                     transform=test_tsfm)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

        checkpoint = torch.load(path + filename)
        args.arch = checkpoint['arch']

        # Creates model using EfficientNet library
        if 'efficientnet-b' in args.arch:  # NEW
            if args.pretrained:
                model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop, num_classes=num_classes)
                print("=> using pre-trained model '{}'".format(args.arch))
            else:
                print("=> creating model '{}'".format(args.arch))
                model = EfficientNet.from_name(args.arch, override_params={'num_classes': num_classes})

        # Handles the case where the user has specified GPU, to set the device manually
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # If GPU not specifiec check if CUDA is available, else use CPU
            if torch.cuda.is_available():
                model = torch.nn.DataParallel(model).cuda()
            else:
                model.to("cpu")

        # args.start_epoch = checkpoint['epoch']
        # best_acc1 = checkpoint['best_acc1']

        # Loads model from loaded state dictionary in checkpoint
        #print(checkpoint['state_dict'].keys())
        model.load_state_dict(checkpoint['state_dict'])

        # creates and loads optimizer
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    nesterov=True,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Defines criterion for model
        if torch.cuda.is_available():
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        else:
            criterion = nn.CrossEntropyLoss()

        # Enables the inbuilt cudnn auto-tuner to find the best algorithm to use for the hardware
        cudnn.benchmark = True

        y_pred, y_true, p_pred = getpred(test_loader, model, criterion, args, 'Test: ')

        get_reporting(y_pred, y_true)

        # Checks if y_true is the same in for all filters
        if y_pred_per_filter["True"]:
            assert y_pred_per_filter["True"] == y_true
        else:
            y_pred_per_filter["True"] = y_true

        # Saves the prediction in the dictionary
        if filter:
            y_pred_per_filter[filter] = y_pred
        else:
            y_pred_per_filter["No_filter"] = y_pred

        # Saves the prediction probabilites in the dictionary
        if filter:
            p_pred_per_filter[filter] = p_pred
        else:
            p_pred_per_filter["No_filter"] = p_pred

    print("Ensemble Top 1 majority voting:")
    get_reporting(ensemble_majority_voting(y_pred_per_filter), y_true)
    f = open( args.outdest + "top1_majority_voting_predictions.txt", "w")
    f.write(str(y_pred_per_filter))
    f.close()

    print("Ensemble Pontalba majority voting:")
    get_reporting(ensemble_pontalba(p_pred_per_filter), y_true)
    f = open(args.outdest + "pontalba_majority_voting_predictions.txt", "w")
    f.write(str(p_pred_per_filter))
    f.close()

    save_reporting(y_pred, y_true, filter)


def ensemble_majority_voting(y_pred_per_filter):
    all = [y_pred_per_filter["No_filter"], y_pred_per_filter["_Macenko"], y_pred_per_filter["_Reinhard"], y_pred_per_filter["_SCD"]]
    all = np.array(all).transpose()
    return [np.random.choice(np.flatnonzero(np.bincount(v) == np.bincount(v).max())) for v in all]

def ensemble_pontalba(p_pred_per_filter):
    all = np.array([p_pred_per_filter["No_filter"], p_pred_per_filter["_Macenko"], p_pred_per_filter["_Reinhard"], p_pred_per_filter["_SCD"]])
    all = np.sum(all, axis=0) / all.shape[0]
    return [v.argmax() for v in all]


def get_reporting(y_pred,y_true):
    print("classification report:")
    print(classification_report(y_true, y_pred))
    print(confusion_matix_string(y_true, y_pred))

def save_reporting(y_pred,y_true,filter):
    f = open(args.outdest + "classification_report.txt", "w")
    f.write('-------',str(filter),'-------')
    f.write(str(classification_report(y_true, y_pred, output_dict=True)))
    f.close()


def getpred(val_loader, model, criterion, args, testing_type):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    topC1 = AverageMeter('Acc C1', ':6.2f')
    topC2 = AverageMeter('Acc C2', ':6.2f')
    topC3 = AverageMeter('Acc C3', ':6.2f')
    topC4 = AverageMeter('Acc C4', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, topC1, topC2, topC3, topC4,
                             prefix=testing_type)

    # switch to evaluate mode
    model.eval()

    y_true, y_pred = [], []
    p_pred = []
    # Source https://github.com/DingXiaoH/RepVGG/blob/main/train.py
    with torch.no_grad(): #All new tensors do not require gradient (requires_grad=False)
        end = time.time()
        # iterates over all batches
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()

            # extend y_true and y_pred with the prediction and true class
            y_true.extend(target.detach().cpu().numpy())
            y_pred.extend(pred.detach().cpu().numpy()[0])

            # extend p_pred with the softmaxed predictions accuracies
            softmax = nn.Softmax(dim=1)
            p_pred.extend(softmax(output).detach().cpu().numpy())

            C1indices = [index for index, element in enumerate(target) if element == 0]
            if len(C1indices) > 0:
                accC1 = accuracy(output[C1indices], target[C1indices], topk=(1,))
                topC1.update(accC1[0].item(), len(C1indices))

            C2indices = [index for index, element in enumerate(target) if element == 1]
            if len(C2indices) > 0:
                accC2 = accuracy(output[C2indices], target[C2indices], topk=(1,))
                topC2.update(accC2[0].item(), len(C2indices))

            C3indices = [index for index, element in enumerate(target) if element == 2]
            if len(C3indices) > 0:
                accC3 = accuracy(output[C3indices], target[C3indices], topk=(1,))
                topC3.update(accC3[0].item(), len(C3indices))

            C4indices = [index for index, element in enumerate(target) if element == 3]
            if len(C4indices) > 0:
                accC4 = accuracy(output[C4indices], target[C4indices], topk=(1,))
                topC4.update(accC4[0].item(), len(C4indices))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if i % args.print_freq == 0:
            #   progress.print(i)

        #print(' * Acc {top1.avg:.3f}'.format(top1=top1))

    return y_pred, y_true, p_pred




if __name__ == '__main__':
    main()
