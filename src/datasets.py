"""
datasets.py
-----------
MNIST 및 CIFAR-10 데이터셋 로딩 유틸리티.
torchvision을 사용해 데이터를 다운로드하고, DataLoader를 반환한다.
"""

from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 프로젝트 루트 기준 data 디렉토리 (src/ 의 상위)
_DATA_ROOT = str(Path(__file__).parent.parent / "data")

# MNIST 채널별 mean/std
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)

# CIFAR-10 채널별 mean/std (ImageNet-style normalization)
# 출처: https://github.com/kuangliu/pytorch-cifar
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_mnist_dataloaders(batch_size=64, num_workers=2):
    """MNIST 데이터셋 로더 반환"""
    # 주의: Normalize 적용 시 픽셀 범위가 [0,1]을 벗어남
    # → 공격 함수(fgsm.py, pgd.py)에서 clamp 범위를 정규화 공간 기준으로 처리하거나
    #   denormalize_mnist() 후 [0,1]에서 처리해야 함
    transform = transforms.Compose([
        transforms.ToTensor(),              # [0,255] → [0.0,1.0]
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])

    train_dataset = datasets.MNIST(root=_DATA_ROOT, train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=_DATA_ROOT, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

def denormalize_mnist(x: torch.Tensor) -> torch.Tensor:
    """정규화된 MNIST 텐서를 [0, 1] 범위로 되돌린다."""
    mean = torch.tensor(MNIST_MEAN, device=x.device).view(1, 1, 1)
    std  = torch.tensor(MNIST_STD,  device=x.device).view(1, 1, 1)
    return (x * std + mean).clamp(0, 1)

def get_cifar10_dataloaders(batch_size: int = 64, num_workers: int = 2):
    """
    CIFAR-10 데이터셋 로더 반환
    """
    train_transform = transforms.Compose([
        # 데이터 증강: 과제 요구사항(≥80% 정확도) 달성을 위해 추가
        # RandomCrop: 이미지 주변에 4px padding 후 32x32로 무작위 크롭
        transforms.RandomCrop(32, padding=4),
        # RandomHorizontalFlip: 50% 확률로 좌우 반전
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalize: 채널별 mean/std로 정규화 → 학습 안정성 및 정확도 향상
        # 정규화 후 픽셀 범위는 [0,1]이 아님에 주의
        # 공격 함수에서 clamp할 때 정규화 공간 기준으로 처리하거나,
        # 역정규화(denormalize_cifar10) 후 [0,1]에서 처리해야 함
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    # 테스트/평가 시에는 데이터 증강 없이 정규화만 적용
    # (증강은 학습 다양성을 위한 것이므로 평가에서는 불필요)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=_DATA_ROOT, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=_DATA_ROOT, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader

def denormalize_cifar10(x: torch.Tensor) -> torch.Tensor:
    """CIFAR-10 정규화된 텐서를 원래 픽셀 범위로 되돌리는 함수"""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1).to(x.device)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1).to(x.device)
    return (x * std + mean).clamp(0, 1)