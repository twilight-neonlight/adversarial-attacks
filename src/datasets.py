"""
datasets.py
-----------
MNIST 및 CIFAR-10 데이터셋 로딩 유틸리티.
torchvision을 사용해 데이터를 다운로드하고, DataLoader를 반환한다.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10 채널별 mean/std (ImageNet-style normalization)
# 출처: https://github.com/kuangliu/pytorch-cifar
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# MNIST 채널별 mean/std
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)

def get_mnist_dataloaders(batch_size=64, num_workers=2): 
    """MNIST 데이터셋 로더 반환"""
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,255] -> [0.0,1.0]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def get_cifar10_dataloaders(batch_size=64, num_workers=2):
    """CIFAR-10 데이터셋 로더 반환"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def denomalize_cifar10