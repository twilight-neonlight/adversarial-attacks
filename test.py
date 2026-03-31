"""
test.py
-------
학습 및 공격 실행 스크립트.

실행 순서:
  1. MNIST / CIFAR-10 모델 학습 (또는 기존 가중치 로드)
  2. 각 공격 방법(FGSM targeted/untargeted, PGD targeted/untargeted) 실행
  3. 공격 성공률 출력 및 시각화 저장 (results/)

실행 방법:
  python test.py
"""

import functools
from pathlib import Path

import torch

from src.datasets import (
    get_mnist_dataloaders,
    get_cifar10_dataloaders,
    denormalize_mnist,
    denormalize_cifar10,
)
from src.models import MNISTClassifier, ResNet18
from src.train import train_mnist, train_cifar10, MNIST_MODEL_PATH, CIFAR10_MODEL_PATH
from src.fgsm import fgsm_targeted, fgsm_untargeted
from src.pgd import pgd_targeted, pgd_untargeted
from src.utils import evaluate_attack, visualize_attack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────────

# 과제 요구: ε ∈ {0.05, 0.1, 0.2, 0.3} 표 작성용
EPS_LIST    = [0.05, 0.1, 0.2, 0.3]
EPS_DEFAULT = 0.3       # 시각화 및 단일 성공률 출력에 사용할 기본 eps

# PGD 하이퍼파라미터 (과제 권장값)
PGD_K        = 40       # 반복 횟수
PGD_EPS_STEP = 0.01     # 스텝 크기

N_SAMPLES    = 100      # 성공률 계산 샘플 수 (과제 요구: 최소 100)
N_VIS        = 5        # 시각화 샘플 수 (과제 요구: 최소 5)


# ── 모델 로드 헬퍼 ─────────────────────────────────────────────────────────

def load_or_train_mnist():
    """저장된 가중치가 있으면 로드, 없으면 학습 후 반환."""
    model = MNISTClassifier().to(DEVICE)
    if Path(MNIST_MODEL_PATH).exists():
        print(f"[MNIST] 저장된 가중치 로드: {MNIST_MODEL_PATH}")
        model.load_state_dict(torch.load(MNIST_MODEL_PATH, map_location=DEVICE))
        model.eval()
    else:
        print("[MNIST] 저장된 가중치 없음 → 학습 시작")
        model = train_mnist()
    return model


def load_or_train_cifar10():
    """저장된 가중치가 있으면 로드, 없으면 학습 후 반환."""
    model = ResNet18().to(DEVICE)
    if Path(CIFAR10_MODEL_PATH).exists():
        print(f"[CIFAR-10] 저장된 가중치 로드: {CIFAR10_MODEL_PATH}")
        model.load_state_dict(torch.load(CIFAR10_MODEL_PATH, map_location=DEVICE))
        model.eval()
    else:
        print("[CIFAR-10] 저장된 가중치 없음 → 학습 시작")
        model = train_cifar10()
    return model


# ── 공격 실행 헬퍼 ─────────────────────────────────────────────────────────

def run_attacks(model, test_loader, dataset_name, denorm_fn):
    """
    4가지 공격을 순서대로 실행한다.
    - 각 공격마다 eps별 성공률 표 출력
    - EPS_DEFAULT 기준 시각화 저장
    """
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()} 공격 시작")
    print(f"{'='*60}")

    # 공격별 설정: (공격명, 함수, targeted 여부)
    attacks = [
        ("fgsm_targeted",   fgsm_targeted,   True),
        ("fgsm_untargeted", fgsm_untargeted, False),
        ("pgd_targeted",    pgd_targeted,    True),
        ("pgd_untargeted",  pgd_untargeted,  False),
    ]

    # eps별 성공률 표 (report.pdf 작성용)
    print(f"\n  {'공격':<20} | " + " | ".join(f"eps={e}" for e in EPS_LIST))
    print(f"  {'-'*20}-+-" + "-+-".join("-"*7 for _ in EPS_LIST))

    for attack_name, attack_fn, is_targeted in attacks:
        row = f"  {attack_name:<20} |"
        for eps in EPS_LIST:
            # functools.partial로 model과 eps를 고정한 공격 함수 생성
            if attack_name.startswith("fgsm"):
                fn = functools.partial(attack_fn, model, epsilon=eps)
            else:
                # PGD: eps_step은 고정, eps만 변경
                fn = functools.partial(attack_fn, model,
                                       k=PGD_K, eps=eps, eps_step=PGD_EPS_STEP)

            rate = evaluate_attack(
                model, fn, test_loader, DEVICE,
                n_samples=N_SAMPLES,
                targeted=is_targeted,
            )
            row += f" {rate:6.2f}% |"
        print(row)

    # EPS_DEFAULT 기준 시각화 저장
    print(f"\n  시각화 저장 중 (eps={EPS_DEFAULT})...")
    for attack_name, attack_fn, is_targeted in attacks:
        if attack_name.startswith("fgsm"):
            fn = functools.partial(attack_fn, model, epsilon=EPS_DEFAULT)
        else:
            fn = functools.partial(attack_fn, model,
                                   k=PGD_K, eps=EPS_DEFAULT, eps_step=PGD_EPS_STEP)

        visualize_attack(
            model, fn, test_loader, DEVICE,
            dataset_name=dataset_name,
            attack_name=attack_name,
            targeted=is_targeted,
            n_samples=N_VIS,
            denorm_fn=denorm_fn,
        )


# ── 메인 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 데이터 로더
    _, mnist_test_loader   = get_mnist_dataloaders()
    _, cifar10_test_loader = get_cifar10_dataloaders()

    # 2. 모델 로드 또는 학습
    mnist_model   = load_or_train_mnist()
    cifar10_model = load_or_train_cifar10()

    # 3. 공격 실행
    run_attacks(mnist_model,   mnist_test_loader,   "mnist",   denormalize_mnist)
    run_attacks(cifar10_model, cifar10_test_loader, "cifar10", denormalize_cifar10)

    print("\n모든 공격 완료. 결과는 results/ 디렉토리를 확인하세요.")