"""
utils.py
--------
공격 결과 평가 및 시각화 유틸리티.

- evaluate_attack : 공격 성공률(%) 계산
- visualize_attack: 원본/adversarial/perturbation 나란히 시각화 후 results/에 저장
"""

from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 화면 출력 없이 파일로 저장 (서버/CLI 환경 대응)

from src.datasets import denormalize_mnist, denormalize_cifar10

# 프로젝트 루트 기준 results 디렉토리 (src/ 의 상위)
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = str(RESULTS_DIR)

# CIFAR-10 클래스 이름 (시각화 레이블용)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── 공격 성공률 ────────────────────────────────────────────────────────────

def evaluate_attack(model, attack_fn, loader, device, n_samples=100, targeted=False):
    """
    공격 성공률(%) 계산.

    targeted=True일 때 target은 (label + 1) % 10으로 자동 설정.
    → 항상 정답과 다른 클래스 보장, 고정 target보다 공정한 평가 가능.

    Returns:
        success_rate: 공격 성공률 (0.0 ~ 100.0)
    """
    model.eval()
    success = 0
    total   = 0

    for images, labels in loader:
        for i in range(images.size(0)):
            if total >= n_samples:
                break

            x     = images[i].unsqueeze(0).to(device)  # [1, C, H, W]
            label = labels[i].unsqueeze(0).to(device)  # [1]

            if targeted:
                # (label + 1) % 10: 항상 정답과 다른 클래스로 유도
                target_class = (labels[i].item() + 1) % 10
                y_target = torch.tensor([target_class], device=device)
                x_adv    = attack_fn(x, y_target)
            else:
                # untargeted: 정답 클래스가 아닌 다른 클래스로 오분류 여부 확인
                x_adv = attack_fn(x, label)

            # 공격 후 예측
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)

            if targeted:
                # targeted 성공: target 클래스로 예측
                success += (pred.item() == target_class)
            else:
                # untargeted 성공: 정답 클래스가 아닌 다른 클래스로 예측
                success += (pred.item() != labels[i].item())

            total += 1

        if total >= n_samples:
            break

    return success / total * 100

# ── 공격 시각화 ───────────────────────────────────────────────────────────

def visualize_attack(model, attack_fn, loader, device, dataset_name,
                     attack_name, targeted=False,
                     n_samples=5, denorm_fn=None):
    """
    원본 이미지, adversarial 이미지, perturbation을 나란히 시각화하여 저장.
    각 샘플에 대해 원본/adversarial/perturbation을 나란히 표시한다.

    targeted=True일 때 target은 (label + 1) % 10으로 자동 설정.
    """
    model.eval()
    collected = 0

    for images, labels in loader:
        for i in range(images.size(0)):
            if collected >= n_samples:
                break

            x     = images[i].unsqueeze(0).to(device)
            label = labels[i].unsqueeze(0).to(device)

            if targeted:
                # (label + 1) % 10: 항상 정답과 다른 클래스로 유도
                target_class = (labels[i].item() + 1) % 10
                y_target = torch.tensor([target_class], device=device)
                x_adv    = attack_fn(x, y_target)
            else:
                x_adv = attack_fn(x, label)

            # 공격 전/후 예측
            with torch.no_grad():
                pred_orig = model(x).argmax(dim=1).item()
                pred_adv  = model(x_adv).argmax(dim=1).item()

            # 시각화를 위해 역정규화 후 CPU로 이동
            x_vis     = denorm_fn(x.cpu()).squeeze(0)     if denorm_fn else x.cpu().squeeze(0)
            x_adv_vis = denorm_fn(x_adv.cpu()).squeeze(0) if denorm_fn else x_adv.cpu().squeeze(0)

            # perturbation 시각화: 차이를 magnify해서 보기 쉽게
            # abs()로 음수 제거, *10으로 확대 후 clamp(0,1)
            perturbation = (x_adv_vis - x_vis).abs() * 10
            perturbation = perturbation.clamp(0, 1)

            # 레이블 텍스트 설정
            if dataset_name == "cifar10":
                orig_label = CIFAR10_CLASSES[labels[i].item()]
                orig_pred  = CIFAR10_CLASSES[pred_orig]
                adv_pred   = CIFAR10_CLASSES[pred_adv]
            else:
                orig_label = str(labels[i].item())
                orig_pred  = str(pred_orig)
                adv_pred   = str(pred_adv)

            # 3열 subplot: 원본 / adversarial / perturbation
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))

            # MNIST: (1, H, W) → (H, W), CIFAR-10: (3, H, W) → (H, W, 3)
            def to_numpy(t):
                t = t.numpy()
                return t.squeeze() if dataset_name == "mnist" else t.transpose(1, 2, 0)

            axes[0].imshow(to_numpy(x_vis),          cmap="gray" if dataset_name == "mnist" else None)
            axes[0].set_title(f"Original\nTrue: {orig_label}\nPred: {orig_pred}")
            axes[0].axis("off")

            axes[1].imshow(to_numpy(x_adv_vis),      cmap="gray" if dataset_name == "mnist" else None)
            axes[1].set_title(f"Adversarial\nPred: {adv_pred}")
            axes[1].axis("off")

            axes[2].imshow(to_numpy(perturbation),   cmap="gray" if dataset_name == "mnist" else None)
            axes[2].set_title("Perturbation (×10)")
            axes[2].axis("off")

            plt.tight_layout()

            # results/ 디렉토리에 저장
            fname = f"{RESULTS_DIR}/{dataset_name}_{attack_name}_{collected+1}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()

            collected += 1

        if collected >= n_samples:
            break

    print(f"  시각화 저장 완료: {RESULTS_DIR}/{dataset_name}_{attack_name}_1~{n_samples}.png")

    