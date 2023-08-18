import argparse
import os
import random
import shutil
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from model_Resnet50 import CPML
from tqdm import tqdm
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#######################
##### 1 - Setting #####
#######################

##### args setting
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', default='./datasets/FGVC_Aircraft', help='dataset dir')
parser.add_argument('-b', '--batch_size', default=64, help='batch_size')
parser.add_argument(
    '-g', '--gpu', default='0', help='example: 0 or 1, to use different gpu'
)
parser.add_argument('-w', '--num_workers', default=8, help='num_workers of dataloader')
parser.add_argument('-s', '--seed', default=2020, help='random seed')
parser.add_argument(
    '-n',
    '--note',
    default='',
    help='exp note, append after exp folder, fgvc(_r50) for example',
)
parser.add_argument(
    '-a',
    '--amp',
    default=0,
    help='0: w/o amp, 1: w/ nvidia apex.amp, 2: w/ torch.cuda.amp',
)
args = parser.parse_args()


##### exp setting
seed = int(args.seed)
datasets_dir = args.dir
nb_epoch = 128  # 128 as default to suit scheduler
batch_size = int(args.batch_size)
num_workers = int(args.num_workers)
lr_begin = (batch_size / 256) * 0.1  # learning rate at begining
use_amp = int(args.amp)  # use amp to accelerate training


##### dataset settings
# data_dir = join('dataset', datasets_dir)
data_dir = datasets_dir
data_sets = ['train', 'test']
nb_class = len(
    os.listdir(join(data_dir, data_sets[0]))
)  # get number of class via img folders automatically
exp_dir = 'result/{}{}'.format(datasets_dir, args.note)  # the folder to save model


##### CUDA device setting
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


##### Random seed setting
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


##### Dataloader setting
re_size = 512
crop_size = 448

train_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.RandomCrop(crop_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

train_set = ImageFolder(root=join(data_dir, data_sets[0]), transform=train_transform)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
)


##### Model settings
net = CPML(num_classes=nb_class)
net = net.to(device)

for param in net.parameters():
    param.requires_grad = True  # make parameters in model learnable


##### optimizer setting
optimizer = torch.optim.SGD(
    net.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, eta_min=0.0001)


##### file/folder prepare
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

with open(os.path.join(exp_dir, 'train_log_resnet50_CPML.csv'), 'w+') as file:
    file.write('Epoch, lr, Train_Loss, Train_Acc, Test_Acc\n')


min_train_loss = float('inf')
max_eval_acc = 0
loss_function = nn.CrossEntropyLoss().to(device)
loss_kl = nn.KLDivLoss(reduction='batchmean').to(device)
best_acc = 0

for epoch in range(nb_epoch):
    print('\n===== Epoch: {} ====='.format(epoch))
    net.train()  # set model to train mode, enable Batch Normalization and Dropout
    lr_now = optimizer.param_groups[0]['lr']
    train_loss = train_correct = train_total = idx = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, ncols=80)):
        idx = batch_idx

        if inputs.shape[0] < batch_size:
            continue

        optimizer.zero_grad()  # Sets the gradients to zero
        inputs, targets = inputs.to(device), targets.to(device)

        x, logits_r = net(inputs, flag="train")
        loss = loss_function(x, targets) + loss_function(logits_r, targets) + (0.1 * (loss_kl(F.log_softmax(x, dim=1), F.softmax(logits_r, dim=1)) + loss_kl(F.log_softmax(logits_r, dim=1), F.softmax(x, dim=1))))
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(x.data, 1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets.data).cpu().sum()
        train_loss += loss.item()

    scheduler.step()

    train_acc = 100.0 * float(train_correct) / train_total
    train_loss = train_loss / (idx + 1)
    print(
        'Train | lr: {:.4f} | Loss: {:.4f} | Acc: {:.3f}% ({}/{})'.format(
            lr_now, train_loss, train_acc, train_correct, train_total
        )
    )

    ##### Evaluating model with test dataset every epoch
    with torch.no_grad():
        net.eval()  # set model to eval mode, disable Batch Normalization and Dropout
        eval_set = ImageFolder(
            root=join(data_dir, data_sets[-1]), transform=test_transform
        )
        eval_loader = DataLoader(
            eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        eval_correct = eval_total = 0
        for _, (inputs, targets) in enumerate(tqdm(eval_loader, ncols=80)):
            inputs, targets = inputs.to(device), targets.to(device)
            val_outputs, _ = net(inputs, flag="val")
            _, predicted = torch.max(val_outputs.data, 1)
            eval_total += targets.size(0)
            eval_correct += predicted.eq(targets.data).cpu().sum()
        eval_acc = 100.0 * float(eval_correct) / eval_total
        if eval_acc > best_acc:
            best_acc = eval_acc
        print(
            '{} | Acc: {:.3f}% ({}/{})'.format(
                data_sets[-1], eval_acc, eval_correct, eval_total
            )
        )

        ##### Logging
        with open(os.path.join(exp_dir, 'train_log_resnet50_CPML.csv'), 'a+') as file:
            file.write(
                '{}, {:.4f}, {:.4f}, {:.3f}%, {:.3f}%\n'.format(
                    epoch, lr_now, train_loss, train_acc, eval_acc
                )
            )

        ##### save model with highest acc
        if eval_acc > max_eval_acc:
            max_eval_acc = eval_acc
            torch.save(
                net.state_dict(),
                os.path.join(exp_dir, 'max_acc_resnet50_CPML.pth'),
                _use_new_zipfile_serialization=False,
            )

print("*********best_acc: ", best_acc, " ***********")