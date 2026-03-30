"""
utils.py
--------
공격 결과 평가 및 시각화 유틸리티.

- evaluate_attack : 공격 성공률(%) 계산
- visualize_attack: 원본/adversarial/perturbation 나란히 시각화 후 results/에 저장
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 화면 출력 없이 파일로 저장 (서버/CLI 환경 대응)

from src.datasets import denormalize_mnist, denormalize_cifar10

# 결과 저장 디렉토리 (과제 요구사항)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# CIFAR-10 클래스 이름 (시각화 레이블용)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

