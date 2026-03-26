"""
models.py
---------
MNIST 및 CIFAR-10 분류기 모델 정의.

CIFAR-10 모델(ResNet18) 구조 출처:
    https://github.com/kuangliu/pytorch-cifar
    (MIT License)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F