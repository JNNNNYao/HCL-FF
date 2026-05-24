import copy
import json

import torch

from torchvision import datasets, transforms

from .build_hierarchy import json_to_hierarchy


def get_cifar10_dataloader(root, train_batch_size=128, test_batch_size=128, seed=2222):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.4, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)

    with open("./data/hierarchy_cifar10.json", "r") as f:
        json_data = json.load(f)
    aliases = {
        "car": "automobile",
    }
    hierarchy = json_to_hierarchy(
        graph_json=json_data,
        class_order=train_dataset.classes,
        label_aliases=aliases
    )

    torch.manual_seed(seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])
    
    valid_dataset = copy.deepcopy(valid_dataset)
    valid_dataset.dataset.transform = transform

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=8
    )

    return train_loader, valid_loader, test_loader, hierarchy


