# Find all differences here https://www.diffchecker.com/PlnW76qN
# imports
import argparse
import os
import random
import shutil
import time
import warnings
import PIL
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from efficientnet_pytorch import EfficientNet
from CustomDataset import CustomDataset
import matplotlib.pyplot as plt

"""
Name: run_model
Function: Handles if user specified seed or GPU and warns the user of consequences (process slowdown)
Input: loaders (dict with three dataLoader objects), split (current split number in k crossfold validation), args (arguments from main) and class_names
Output: main_worker with same input
Source: https://github.com/pytorch/examples/blob/master/imagenet/main.py
Source2: https://www.xilinx.com/html_docs/vitis_ai/1_3/pytorch_ex.html
"""
num_classes = 4
best_all_splits_acc = 0
def run_model(loaders, split, args, class_names,num_item_per_class):
    # Uses seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Warns user if they chose specific GPU, and prints out number of available GPUs
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    ngpus = torch.cuda.device_count()
    print("Available GPUs: {}".format(ngpus))

    return main_worker(loaders, split, args.gpu, ngpus, args, num_item_per_class)

"""
Type: function
Name: main_worker
Function: After run_model we have handled some manual config (seed or GPU), this function creates a model according to
        passed arguments and trains/validates/tests it on the current data in the dataloader.
        This function drives the whole training, validation and testing process.
Input: Data loader (dictionary), current split (k), gpu, number of gpus, args, number of items in each class
"""
def main_worker(loaders, split, gpu, ngpus, args, num_item_per_class):
    global best_all_splits_acc
    best_acc1 = 0
    args.gpu = gpu
    output_destination = args.outdest

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Creates model using EfficientNet library
    if 'efficientnet-b' in args.arch:  # NEW
        if args.pretrained:
            model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop, num_classes=num_classes)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(args.arch, override_params={'num_classes': num_classes})

    else:
        print("Only EfficientNet models are supported.")
        quit()

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

    # Class weights = (total_count - class_count) / total_count
    # old weights:
    # weights = [25137/2017, 25137/3211, 25137/7296, 25137/12519]
    # weights = [(25137-2017)/25137, (25137-3211)/25137, (25137-7296)/25137, (25137-12519)/25137]
    # Our changes:
    total_num_train = sum(num_item_per_class)
    weights = [(total_num_train-num)/total_num_train for num in num_item_per_class]
    class_weights = torch.FloatTensor(weights)
    if torch.cuda.is_available():
        class_weights = class_weights.cuda()

    # Define loss function (criterion)
    # Cross Entropy loss is a combination of LogSoftMax and NLLLoss i one single class
    # changes made to if statement structure
    if torch.cuda.is_available():
        if args.upsample:
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(args.gpu)
    else:
        if args.upsample:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Defines Stocastic Gradient Descent Optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                nesterov=True,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), args.lr,weight_decay=args.weight_decay, amsgrad=True)

    # Resume from a checkpoint is args.resume is not an empty string and a valid path
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Eenables the inbuilt cudnn auto-tuner to find the best algorithm to use for the hardware
    cudnn.benchmark = True

    # Unfreezes only the classification layer if feature exract argument is set to True
    # param.requires_grad = True -> unfreezed
    # param.requires_grad = False -> freezed
    if args.feature_extract:
        c = 0
        for param in model.parameters():
            c+= 1
            if c < 213:
                param.requires_grad = False

    # Defines losses and accuracy to later be used for plotting the loss and accuracy curves
    epochs = [epoch for epoch in range(args.start_epoch, args.epochs)]
    losses = []
    accs = []

    # Trains and validate in all epochs
    for epoch in epochs:
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(loaders['train'], model, criterion, optimizer, epoch, args, losses, accs)

        # evaluate on validation set
        acc1, val_classification, val_confusion_matrix = validate(loaders['val'], model, criterion, args,'Validation: ')

        # remember best acc@1
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # remember best acc@1 over all k splits
        is_best_all = acc1 > best_all_splits_acc
        best_all_splits_acc = max(acc1, best_all_splits_acc)

        # Saves checkpoint of current model see function comment
        save_checkpoint({
            'epoch': epoch + 1,# if we use resume (args.resume) we should start from the next epoch
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, is_best_all, split, output_destination)

        # Classifies current validation set and writes statistics to correct file
        if args.validate:
            if is_best:
                with open(output_destination + '/res_val_{}.txt'.format(split), 'w') as f:
                    print("epoch:", epoch, file = f)
                    print(acc1, file=f)
                    print(val_classification, file=f)
                    print(val_confusion_matrix, file=f)
            if is_best_all:
                with open(output_destination + '/res_val.txt', 'w') as f:
                    print("split:", split, file = f)
                    print("epoch:", epoch, file = f)
                    print(acc1, file=f)
                    print(val_classification, file=f)
                    print(val_confusion_matrix, file=f)

        # Classifies testing set and writes statistics to correct file
        if args.evaluate:
            if is_best:
                test_acc1, test_classification, test_confusion_matrix = validate(loaders['test'], model, criterion, args, 'Test: ')
                with open(output_destination + '/res_test_{}.txt'.format(split), 'w') as f:
                    print("epoch:", epoch, file=f)
                    print(test_acc1, file=f)
                    print(test_classification, file=f)
                    print(test_confusion_matrix, file=f)
            if is_best_all:
                with open(output_destination + '/res_test.txt', 'w') as f:
                    print("split:", split, file = f)
                    print("epoch:", epoch, file = f)
                    print(test_acc1, file=f)
                    print(test_classification, file=f)
                    print(test_confusion_matrix, file=f)

    with open(output_destination + '/losses_vs_epochs_{}.txt'.format(split), 'w') as f:
        print(losses, file = f)
        print(epochs, file = f)
    with open(output_destination + '/acc_vs_epochs_{}.txt'.format(split), 'w') as f:
        print(accs, file = f)
        print(epochs, file = f)

"""
Name: train
Function: trains the model with the given loader, model, criterion and optimizer
Output: Creates print log in terminal for progress and average for
Computational time, Data loading time,Loss, Acc@1.
Epoch: [epoch number] [Bacth count/Number of batches in epoch]
Each epoch consists of a fixed number of batches depending on args.batch_size parameter
Number of batxhes in epoch = roof(Dataset size (labels) / batch size)

"""
def train(train_loader, model, criterion, optimizer, epoch, args, losses_vec, accs_vec):
    # Epoch and batch count / Time per batch / Data loading time for batch
    batch_time = AverageMeter('Time', ':6.3f') # Training time per batch
    data_time = AverageMeter('Data loading time', ':6.3f') # Data loading time from python iteration computation
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                              prefix="Epoch: [{}]".format(epoch))

    # Switches to train mode for the EfficientNet
    model.train()

    end = time.time()

    running_loss = 0.0
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    losses_vec.append(losses.avg)
    accs_vec.append(top1.avg)
"""
Name: validate
Function: validate or tests the model with the given loader, model, criterion
Output: Creates print log in terminal for progress and average for
Computational time, Data loading time,Loss, Acc@1.
Acc@1 is top 1% accuracy which is the highest accuracy of 4 in the lat classification layer.
For example [0.3,0.2,0.4,0.1] gives class 3  as top 1 class (with aprediction ccuracy 0.4),
this can also be calculatied by taking diagonal elements i confusion matrix divvided by total #
of elements. Acc C1 is the number of data points we predicted correctly from class 1 in data set.
Source: https://github.com/DingXiaoH/RepVGG/blob/main/train.py
"""
def validate(val_loader, model, criterion, args, testing_type):
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

    #Creates arrays for prediciton values
    y_true, y_pred = [], []

    with torch.no_grad(): # skips the gradient calculation over the weights, not changing any weight in the specified layers

        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output (training)
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            # gets top 1 prediction
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()

            y_true.extend(target.detach().cpu().numpy())
            y_pred.extend(pred.detach().cpu().numpy()[0])

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

            if i % args.print_freq == 0:
                progress.print(i)

        print(' * Acc {top1.avg:.3f}'
              .format(top1=top1))

    report = classification_report(y_true, y_pred)
    print("classification report:")
    print(report)
    confusion_matrix = confusion_matix_string(y_true, y_pred)
    print(confusion_matrix)
    return top1.avg, report, confusion_matrix
"""
Name: save_checkpoint
Function: Saves a checkpoint of all model params in the file 'checkpoint.pth.tar' located in the destination folder.
Inculdes epoch, optimizer, architecture, statedict, best_acc1, optimizer
Input: Model parameters (state), is_best (boolean value), destinantion (PATH)
Output: Saves checkpoint as checkpoint.pth.tar', if is_best = True copies it to the right destination with name model_best_{k value in crossvalidation}.pth.tar
Source: https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
def save_checkpoint(state, is_best, is_best_all, split, destination ,filename='/checkpoint.pth.tar'):
    torch.save(state, destination + filename)
    if is_best:
        shutil.copyfile(destination + filename, destination + '/model_best_{}.pth.tar'.format(split))
    if is_best_all:
        shutil.copyfile(destination + filename, destination + '/model_best.pth.tar')

"""
Name: AverageMeter
Function: Creates a custom string (__str__) with specified name and format and computes and stores the average and current value
Inastance variables: Name of stored value and print format. Also, calculated value, avarage, sum and count.
Source: https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
class AverageMeter(object):
    def __init__(self, name, fmt=':f'): # fmt means format, and standard for formatting can be found https://docs.python.org/3/library/string.html#format-specification-mini-language
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

"""
Name: ProgressMeter
Function: Creates a custom string (get_batch_fmtstr) with specified name and format, and given instance variables from AverageMeter objects
Instance variables: Custumized string from AvarageMeter objects, name of object and it's custom format
Source: https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


"""
Name: adjust_learning_rate
Function: Sets the learning rate to the initial learning rate decayed by 3% every 2.4 epochs (assuming we learn more in the beginning)
Input: Optimizer, epoch number and specified learning rate (args.lr)
Output: Updating learning rate in model for next epoch
Source: https://gist.github.com/zachguo/10296432
"""
def adjust_learning_rate(optimizer, epoch, input_learning_rate):
    learning_rate = input_learning_rate * (0.97 ** (epoch // 2.4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

"""
Name: accuracy
Function: Computes the accuracy over the k top predictions for the specified values of k"
Input: predicted and target and tuple containing wanted accuracies
Output: Calculate accuracy for print log for each epoch printed
Source: https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py
"""
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


"""
Name: print_cm
Function: Prints confusion matrix in terminal (labels are hardcoded)
Input: True and predicted labels
Output: print in  terminal
Source: https://gist.github.com/zachguo/10296432
"""
def confusion_matix_string(y_true, y_pred, labels= [0,1,2,3],hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    cm_string = "confusion matrix:\n"
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    """pretty print for confusion matrixes"""
    columnwidth = max([len(str(x)) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    cm_string += empty_cell + " t\p "
    for label in labels:
        cm_string += "%{0}s ".format(columnwidth) % label
    cm_string += "\n"
    for i, label1 in enumerate(labels):
        cm_string += "   %{0}s  ".format(columnwidth) % label1
        for j in range(len(labels)):
            cell = "%{0}.0f ".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            cm_string += cell
        cm_string += "\n"
    return cm_string

if __name__ == '__main__':
    main()
