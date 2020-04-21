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
parser.add_argument('--evaluate', '-e', action='store_true', help='Evaluation mode')
parser.add_argument('--epochs', default=1600, type=int, metavar='N',
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

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def uda_train(train_labelled, train_unlabelled, train_unlabelled_aug, model, criterion, consistency_criterion, optimizer, epoch):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()

    lam = 1.0 # TODO Check paper lambda

    train_labelled_iter = iter(train_labelled)
    train_unlabelled_iter = iter(train_unlabelled)
    train_unlabelled_aug_iter = iter(train_unlabelled_aug)

    # measure data loading time
    data_time.update(time.time() - end)

    x, y = next(train_labelled_iter)

    x = x.cuda()
    y = y.cuda()

    y_pred = model(x)

    # Supervised part
    supervised_loss = criterion(y_pred, y)
    writer.add_scalar('Train/Supervised loss', supervised_loss.float())

    # TODO: TSA loss supervised

    # Unsupervised part
    try:
        unsup_x = next(train_unlabelled_iter)
    except StopIteration:
        train_unlabelled_iter = iter(train_unlabelled)
        unsup_x = next(train_unlabelled_iter)

    unsup_x = unsup_x.cuda()
    unsup_orig_y_pred = model(unsup_x).detach()
    unsup_orig_y_probas = torch.softmax(unsup_orig_y_pred, dim=-1)
    consistency_loss1 = consistency_criterion(unsup_orig_y_probas)

    unsup_aug_x = next(train_unlabelled_aug_iter)
    unsup_aug_x = unsup_aug_x.cuda()
    unsup_aug_y_pred = model(unsup_aug_x)
    unsup_aug_y_probas = torch.log_softmax(unsup_aug_y_pred, dim=-1)
    consistency_loss2 = consistency_criterion(unsup_aug_y_probas, unsup_orig_y_probas)

    optimizer.zero_grad()

    final_loss = supervised_loss + lam * (consistency_loss1 + consistency_loss2)
    final_loss.backward()

    optimizer.step()

    # measure accuracy and record loss
    prec1 = accuracy(y_pred.data, y)[0]
    losses.update(final_loss.item(), input.size(0))
    top1.update(prec1.item(), input.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            epoch, 0, len(train_labelled), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1))



def uda_validate(valid_loader, unlabelled_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            writer.add_scalar('Valid/UDA combined loss', loss.float())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(valid_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


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
    train_labelled, train_unlabelled, train_unlabelled_aug, test = dataset.cifar10_unsupervised_dataloaders()

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    consistency_criterion = nn.KLDivLoss(reduction='batchmean').cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=True)

    # warmup steps
    t_max = args.epochs - 120

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # UDA train loop
    for epoch in range(args.start_epoch, args.epochs):

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        uda_train(train_labelled, train_unlabelled, train_unlabelled_aug, model, criterion, consistency_criterion, optimizer, epoch)

        scheduler.step()

        # evaluate on validation set
        prec1 = uda_validate(test, train_unlabelled, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        writer.add_scalar('Acc/valid', best_prec1, epoch)

        # save checkpoint
        if epoch > 0 and epoch % 50 == 0:
            print('Save checkpoint')
            save_checkpoint({
                'epoch': epoch + 1,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        if (is_best):
            print('Saving better model')
            save_checkpoint({
                'epoch': epoch + 1,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, filename=os.path.join(args.save_dir, 'best_model.th'))

            writer.add_scalar('Acc/valid', best_prec1, epoch)

if __name__ == '__main__':
    run_unsupervised()


