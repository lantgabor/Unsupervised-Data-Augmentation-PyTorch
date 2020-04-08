import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

import numpy as np

from torchvision import transforms

np.random.seed(2)


def cifar10_unsupervised_dataloaders():

    print('Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])  # meanstd transformation

    unsupervised_train_transformation = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32, fill=128),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(scale=(0.1, 0.33)),
    ])  # TODO: use for unsupervised

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Train dataset with and without labels
    cifar10_train_ds = datasets.CIFAR10('/data/', transform=transform_train, train='train', download=True)
    # Number of classes calculated here
    num_classes = len(cifar10_train_ds.classes)

    labelled_indices = []
    unlabelled_indices = []

    indices = np.random.permutation(len(cifar10_train_ds))
    class_counters = list([0] * num_classes)
    max_counter = len(cifar10_train_ds) // num_classes

    for i in indices:
        dp = cifar10_train_ds[i]
        if len(cifar10_train_ds) < sum(class_counters):
            unlabelled_indices.append(i)
        else:
            y = dp[1]
            c = class_counters[y]
            if c < max_counter:
                class_counters[y] += 1
                labelled_indices.append(i)
            else:
                unlabelled_indices.append(i)

    # Check
    assert len(set(labelled_indices) & set(unlabelled_indices)) == 0, "{}".format(set(labelled_indices) & set(unlabelled_indices))

    # Labeled and unlabeled dataset
    train_labelled_ds = Subset(cifar10_train_ds, labelled_indices)
    train_unlabelled_ds = Subset(cifar10_train_ds, unlabelled_indices)

    # Data loader for labeled and unlabeled train dataset
    train_labelled = DataLoader(
        train_labelled_ds, batch_size=32, shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    train_unlabelled = DataLoader(
        train_unlabelled_ds, batch_size=32,
        shuffle=False, num_workers=1,
        pin_memory=True
    )

    # Data loader for test dataset
    cifar10_test_ds = datasets.CIFAR10('/data/', transform=transform_test, train='test', download=True)

    test = DataLoader(
        cifar10_test_ds, batch_size=32, shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return train_labelled, train_unlabelled, test, num_classes

def cifar10_supervised_dataloaders():
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True,
            transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]), download=True),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]), download=True),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader

