import copy
import json

import torch

from torchvision import datasets, transforms

from .build_hierarchy import json_to_hierarchy


def get_fmnist_dataloader(root, train_batch_size=128, test_batch_size=128, seed=2222):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])

    # Light augmentation: small rotation + small translation/scale
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])

    train_dataset = datasets.FashionMNIST(root=root, train=True, transform=transform_train, download=True)
    test_dataset = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

    # Load Fashion-MNIST hierarchy
    with open("./data/hierarchy_fashion_mnist.json", "r") as f:
        json_data = json.load(f)
    aliases = {}

    fmnist_order = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    hierarchy = json_to_hierarchy(
        graph_json=json_data,
        class_order=fmnist_order,
        label_aliases=aliases
    )

    torch.manual_seed(seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [55000, 5000])
    
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
