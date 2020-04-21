import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip


np.random.seed(2)

class TransformedDataset1(Dataset):
    def __init__(self, dataset, transform):
        self.data = dataset
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = self.transform(x)
        return x, y

class TransformedDataset2(Dataset):
    def __init__(self, dataset, transform):
        self.data = dataset
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.transform(item)
        return item

def cifar10_unsupervised_dataloaders():
    print('Data Preparation')
    train_transform = Compose([
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # RandomErasing(scale=(0.1, 0.33)),
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    unsupervised_train_transformation = Compose([
        Pad(4),
        RandomCrop(32, fill=128),
        # autoaugment.CIFAR10Policy(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # RandomErasing(scale=(0.1, 0.33)),
    ])

    # Train dataset with and without labels
    cifar10_train_ds = datasets.CIFAR10('/data/', train='train', download=True)
    # Number of classes calculated here
    num_classes = len(cifar10_train_ds.classes)

    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR10.__name__,len(cifar10_train_ds),10))

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


    # Labeled and unlabeled dataset
    train_labelled_ds = Subset(cifar10_train_ds, labelled_indices)
    train_unlabelled_ds = Subset(cifar10_train_ds, unlabelled_indices)

    train_labelled_ds = TransformedDataset1(train_labelled_ds, train_transform)
    train_unlabelled_ds_t = TransformedDataset2(train_unlabelled_ds, train_transform)
    train_unlabelled_aug_ds_t = TransformedDataset2(train_unlabelled_ds, unsupervised_train_transformation)

    # Data loader for labeled and unlabeled train dataset
    train_labelled = DataLoader(
        train_labelled_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    train_unlabelled = DataLoader(
        train_unlabelled_ds_t,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    train_unlabelled_aug = DataLoader(
       train_unlabelled_aug_ds_t,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Data loader for test dataset
    cifar10_test_ds = datasets.CIFAR10('/data/', transform=test_transform, train='test', download=True)

    test = DataLoader(
        cifar10_test_ds, batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_labelled, train_unlabelled, train_unlabelled_aug, test


def cifar10_supervised_dataloaders():
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True,
                         transform=Compose([
                             RandomHorizontalFlip(),
                             RandomCrop(32, 4),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225]),
                         ]), download=True),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR10.__name__,len(train_loader.dataset),10))

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]), download=True),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print('Loading dataset {0} for validating -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR10.__name__,len(val_loader.dataset), 10))

    return train_loader, val_loader
