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

def get_cifar10_dataloaders(batch_size=64, num_workers=2):
    """CIFAR-10 데이터셋 로더 반환"""
    # 학습 시: 데이터 증강 적용 (과제 요구사항 ≥80% 달성을 위해 필요)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # 4px padding 후 32x32 무작위 크롭
        transforms.RandomHorizontalFlip(),       # 50% 확률 좌우 반전
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    # 평가 시: 증강 없이 정규화만 적용
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

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