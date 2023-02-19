import os

import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
from PIL import Image


def build_train_dataloader(path, batch_size, valid_part=0.1, transform=None):
    if transform is None:
        transform = transforms.Compose(
            [transforms.Resize((224,224)), transforms.ToTensor()])

    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    id2label = dataset.classes
    n = len(dataset)
    n_valid = int(valid_part * n)
    valid_dataset = torch.utils.data.Subset(dataset, range(n_valid))  # take first 10%
    train_dataset = torch.utils.data.Subset(dataset, range(n_valid, n))  # take the rest

    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size),
                   'valid': torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)}

    return id2label, dataloaders


def build_predict_dataloader(path, batch_size=4):
    transform = transforms.Resize((224, 224))
    tensors = []
    for img_path in os.listdir(path):
        img = np.array(Image.open(path + img_path))
        img = img / img.max()
        t = torch.tensor(img).permute(2, 1, 0).float()
        tensors.append(transform(t))

    dataloader = torch.utils.data.DataLoader(tensors, batch_size=batch_size)

    return dataloader


def build_cifar_dataloader(batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset.targets = np.array(trainset.targets)
    idx = (trainset.targets == 0) | (trainset.targets == 2) | (trainset.targets == 7) | (trainset.targets == 9)
    trainset.targets = torch.tensor(trainset.targets[idx], dtype=torch.long)
    trainset.data = trainset.data[idx]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testset.targets = np.array(testset.targets)
    idx = (testset.targets == 0) | (testset.targets == 2) | (testset.targets == 7) | (testset.targets == 9)
    testset.targets = torch.tensor(testset.targets[idx], dtype=torch.long)
    testset.data = testset.data[idx]

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    dataloaders = {'train': trainloader, 'valid': testloader}

    id2label = trainset.classes

    print(f"number of classes: {len(id2label)}")
    print(f"train examples: {len(trainset)}, test examples: {len(testset)}")

    return id2label, dataloaders
