import torch
import torchvision
from torchvision.transforms import transforms


def load_data(path, batch_size, valid_part=0.1, transform=None):
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
