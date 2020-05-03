import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing
from RandAugment import RandAugment


np.random.seed(2)

class AddTransform(Dataset):
    def __init__(self, dataset, transform):
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = self.transform(x)
        return x, y

def cifar10_unsupervised_dataloaders():
    print('Data Preparation')
    train_transform = Compose([
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        RandomErasing(scale=(0.1, 0.33)),
    ])

    unsupervised_train_transformation = Compose([
        Pad(4),
        RandomCrop(32, fill=128),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # RANDAUGMENT
    unsupervised_train_transformation.transforms.insert(0, RandAugment(3, 9))

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Train dataset with and without labels
    cifar10_train_ds = datasets.CIFAR10('/data/', train=True, download=True)

    num_classes = len(cifar10_train_ds.classes)

    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR10.__name__,len(cifar10_train_ds),10))

    labelled_indices = []
    unlabelled_indices = []

    indices = np.random.permutation(len(cifar10_train_ds))
    class_counters = list([0] * num_classes)
    max_counter = 10000 // num_classes

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


    # Labelled and unlabelled dataset
    train_labelled_ds = Subset(cifar10_train_ds, labelled_indices)
    train_labelled_ds_t = AddTransform(train_labelled_ds, train_transform)

    # unlabelled ds and aug ds
    train_unlabelled_ds = Subset(cifar10_train_ds, unlabelled_indices)
    train_unlabelled_ds = ConcatDataset([train_unlabelled_ds,train_labelled_ds])

    # apply transformation for both
    train_unlabelled_ds_t = AddTransform(train_unlabelled_ds, train_transform)

    train_unlabelled_aug_ds_t = AddTransform(train_unlabelled_ds, unsupervised_train_transformation)


    print('Labelled dataset -- Num_samples: {0}, classes: {1}, \n Unsupervised dataset -- Num_samples {2}, Augmentation -- Num_samples: {3}'
          .format(len(train_labelled_ds_t), 10, len(train_unlabelled_ds_t), len(train_unlabelled_aug_ds_t)))

    # Data loader for labeled and unlabeled train dataset
    train_labelled = DataLoader(
        train_labelled_ds_t,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    train_unlabelled = DataLoader(
        train_unlabelled_ds_t,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    train_unlabelled_aug = DataLoader(
       train_unlabelled_aug_ds_t,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Data loader for test dataset
    cifar10_test_ds = datasets.CIFAR10('/data/', transform=test_transform, train=False, download=True)

    print('Test set -- Num_samples: {0}'.format(len(cifar10_test_ds)))

    test = DataLoader(
        cifar10_test_ds, batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return train_labelled, train_unlabelled, train_unlabelled_aug, test

def cifar10_supervised_dataloaders(limit = 0):

    if(limit > 0):
        picks = np.random.permutation(limit)

    train_ds = datasets.CIFAR10(root='./data', train=True,
                     transform=Compose([
                         RandomHorizontalFlip(),
                         RandomCrop(32, 4),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
                     ]), download=True)

    if(limit > 0):
        train_ds = Subset(train_ds, picks)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
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
        num_workers=8,
        pin_memory=True
    )

    print('Loading dataset {0} for validating -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR10.__name__,len(val_loader.dataset), 10))

    return train_loader, val_loader
