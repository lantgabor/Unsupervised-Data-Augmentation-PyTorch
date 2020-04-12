import os

import time
import argparse

import torch
import torchvision
from torch import nn, optim
from torch.backends import cudnn
from torch.nn.functional import kl_div, softmax, log_softmax
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import dataset as dataset
import networks

parser = argparse.ArgumentParser(description='Pythorch UDA CIFAR-10 implementation')
parser.add_argument('--supervised_wideresnet', action='store_true', help='Train and eval for baseline')
parser.add_argument('--test', '-t', action='store_true', help='Test mode')
parser.add_argument('--evaluate', '-e', action='store_true', help='Evaluation mode')
parser.add_argument('--baseline', '-b', action='store_true',default=False, help='Train as supervised for baseline')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', '-se', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save', type=str)
parser.add_argument('--resume', default='save', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: save)')
args = parser.parse_args()
best_prec1 = 0

writer =  SummaryWriter()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def uda_train(labelled_loader, unlabelled_loader, valid_loader, num_classes, model, criterion, optimizer, epoch):
    data_time = AverageMeter()

    model.train()

    end = time.time()
    iter_unlabelled = iter(unlabelled_loader)

    for i, (input, target) in enumerate(labelled_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        try:
            unlabel1, unlabel2 = next(iter_unlabelled)
        except StopIteration:
            iter_u = iter(unlabelled_loader)
            unlabel1, unlabel2 = next(iter_u)
        data_all = torch.cat([input, unlabel1, unlabel2]).cuda()

        # supervised
        preds_all = model(data_all)
        preds = preds_all[:len(input)]
        # loss for supervised learning
        loss = criterion(preds, target)

        # unsupervised
        preds_unsup = preds_all[len(input):]
        preds1, preds2 = torch.chunk(preds_unsup, 2)
        preds1 = softmax(preds1, dim=1).detach()
        preds2 = log_softmax(preds2, dim=1)

        loss_kldiv = kl_div(preds2, preds1, reduction='none')
        # loss for unsupervised
        loss_kldiv = torch.sum(loss_kldiv, dim=1)

        loss += 5.0 * torch.mean(loss_kldiv)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



def uda_validate(valid_loader, num_classes, model, criterion):
    pass


def run_unsupervised():
    global args, best_prec1
    best_valid_loss = 10e10

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load model
    model = networks.fastresnet()
    model.cuda()

    # data loaders
    labelled_loader, unlabelled_loader, valid_loader, num_classes = dataset.cifar10_unsupervised_dataloaders()

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    consistency_criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                weight_decay=1e-4)

    # TODO: Change to cosine annealing, gradual warmup
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=args.start_epoch - 1)

    # UDA train loop
    for epoch in range(args.start_epoch, args.epochs):
        # TODO: if RESUME load model
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        uda_train(labelled_loader, unlabelled_loader, valid_loader, num_classes, model, criterion, optimizer, epoch)
        scheduler.step()

        # evaluate on validation set
        prec1 = uda_validate(valid_loader, num_classes, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # TODO: SAVE model




if __name__ == '__main__':
    run_unsupervised()


