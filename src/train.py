"""
train.py
--------
MNIST 및 CIFAR-10 모델 학습 모듈.

학습된 가중치는 프로젝트 루트에 저장된다.
  - mnist_model.pth
  - cifar10_model.pth
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 → 어디서 실행해도 import 경로 일관성 유지
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.datasets import get_mnist_dataloaders, get_cifar10_dataloaders
from src.models import MNISTClassifier, ResNet18

# GPU 사용 가능 시 자동으로 CUDA 선택, 없으면 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습된 가중치 저장 경로 (프로젝트 루트 기준)
ROOT               = Path(__file__).parent.parent
MNIST_MODEL_PATH   = ROOT / "mnist_model.pth"
CIFAR10_MODEL_PATH = ROOT / "cifar10_model.pth"

# ── 공통 헬퍼 ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    """
    한 에폭 학습.
    model.train()으로 dropout/batchnorm을 학습 모드로 전환한다.
    """
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()                    # 이전 배치의 gradient 초기화
        loss = criterion(model(images), labels)  # forward + loss 계산
        loss.backward()                          # gradient 계산
        optimizer.step()                         # 파라미터 업데이트

        running_loss += loss.item()

    # 에폭 평균 loss 반환 (학습 추이 모니터링용)
    return running_loss / len(loader)


def evaluate(model, loader):
    """
    테스트셋 정확도(%) 측정.
    model.eval()로 dropout/batchnorm을 평가 모드로 전환한다.
    torch.no_grad()로 gradient 계산을 비활성화해 메모리/속도 절약.
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds    = model(images).argmax(dim=1)  # 가장 높은 logit의 클래스
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total * 100  # 백분율 반환


# ── MNIST ──────────────────────────────────────────────────────────────────

def train_mnist(epochs=10, save=True):
    """
    MNIST 분류기 학습.

    Args:
        epochs: 최대 학습 에폭 수 (목표 달성 시 조기 종료)
        save  : True이면 학습 완료 후 가중치를 MNIST_MODEL_PATH에 저장

    Returns:
        학습된 MNISTClassifier 모델 (eval 모드)
    """
    print("=" * 50)
    print("MNIST 학습 시작")
    print(f"  Device : {DEVICE}")
    print(f"  목표   : ≥95%")
    print("=" * 50)

    train_loader, test_loader = get_mnist_dataloaders()

    model     = MNISTClassifier().to(DEVICE)
    # Adam: MNIST처럼 단순한 태스크에서 빠른 수렴
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        acc  = evaluate(model, test_loader)

        print(f"  Epoch {epoch:2d}/{epochs} | Loss: {loss:.4f} | Test Acc: {acc:.2f}%")

        # 최고 정확도 갱신 시 가중치 저장 (마지막 에폭이 항상 최선이 아닐 수 있음)
        if acc > best_acc:
            best_acc = acc
            if save:
                torch.save(model.state_dict(), MNIST_MODEL_PATH)

        # 목표 달성 시 조기 종료
        if acc >= 95.0:
            print(f"\n  ✓ 목표 달성: {acc:.2f}% (≥95%) — 조기 종료")
            break
    else:
        # epochs를 모두 소진한 경우
        print(f"\n  최종 정확도: {best_acc:.2f}%")
        if best_acc < 95.0:
            print("  ✗ 목표 미달 — 에폭 수를 늘리거나 모델 구조를 점검할 것")

    if save:
        print(f"  가중치 저장: {MNIST_MODEL_PATH}")

    # 평가 모드로 전환 후 반환 (공격 함수에서 바로 사용 가능하도록)
    model.eval()
    return model


# ── CIFAR-10 ───────────────────────────────────────────────────────────────

def train_cifar10(epochs=30, save=True):
    """
    CIFAR-10 분류기(ResNet18) 학습.

    Args:
        epochs: 최대 학습 에폭 수 (목표 달성 시 조기 종료)
        save  : True이면 학습 완료 후 가중치를 CIFAR10_MODEL_PATH에 저장

    Returns:
        학습된 ResNet18 모델 (eval 모드)
    """
    print("=" * 50)
    print("CIFAR-10 학습 시작")
    print(f"  Device : {DEVICE}")
    print(f"  목표   : ≥80%")
    print("=" * 50)

    train_loader, test_loader = get_cifar10_dataloaders()

    model     = ResNet18().to(DEVICE)
    # SGD + momentum: ResNet 계열에서 Adam보다 최종 정확도가 높은 경향
    # weight_decay: L2 정규화로 과적합 방지
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    # MultiStepLR: epoch 15, 25에서 lr을 0.1배로 감소 → 수렴 안정화
    # 예: 0.1 → 0.01 (epoch 15) → 0.001 (epoch 25)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[15, 25],
        gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()  # lr 스케줄 업데이트 (반드시 optimizer.step() 이후)
        acc  = evaluate(model, test_loader)

        # 현재 lr 출력 (스케줄 확인용)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:2d}/{epochs} | Loss: {loss:.4f} | "
              f"Test Acc: {acc:.2f}% | LR: {current_lr:.4f}")

        # CIFAR-10은 조기 종료 없이 전 에폭 학습
        # best_acc 갱신 시마다 저장 → 최고 성능 가중치 보존
        if acc > best_acc:
            best_acc = acc
            if save:
                torch.save(model.state_dict(), CIFAR10_MODEL_PATH)
            print(f"  ★ 최고 정확도 갱신: {best_acc:.2f}% — 가중치 저장")

    print(f"\n  최종 best 정확도: {best_acc:.2f}%")
    if best_acc < 80.0:
        print("  ✗ 목표 미달 — 에폭 수를 늘리거나 하이퍼파라미터를 점검할 것")
    else:
        print("  ✓ 목표 달성 (≥80%)")

    if save:
        print(f"  가중치 저장: {CIFAR10_MODEL_PATH}")

    model.eval()
    return model


# ── 단독 실행 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # python src/train.py 로 단독 실행 시 두 모델을 순차 학습
    # test.py에서 import할 때는 이 블록이 실행되지 않음
    train_mnist()
    train_cifar10()