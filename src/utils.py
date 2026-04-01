"""
utils.py
--------
공격 결과 평가 및 시각화 유틸리티.

- run_attack: 공격 실행, 성공률 계산, 시각화 저장을 한 번의 루프에서 처리
"""

from pathlib import Path

import torch
from tqdm import tqdm
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


def run_attack(model, attack_fn, loader, device,
               dataset_name, attack_name,
               targeted=False, n_samples=100,
               n_vis=5, denorm_fn=None) -> float:
    """
    공격 실행, 성공률 계산, 시각화 저장을 한 번의 루프에서 처리.

    targeted=True일 때 target은 (label + 1) % 10으로 자동 설정.
    성공 기준:
      - targeted  : pred_adv == (label + 1) % 10
      - untargeted: pred_adv != label

    CPU: 샘플 단위 처리 / GPU: 배치 단위 처리

    Returns:
        success_rate: 공격 성공률 (0.0 ~ 100.0)
    """
    model.eval()
    success       = 0
    total         = 0
    success_cases = []

    if device.type == "cpu":
        # ── CPU: 샘플 단위 처리 ───────────────────────────────────────────
        for images, labels in tqdm(loader, desc=f"  {attack_name}", leave=False):
            for i in range(images.size(0)):
                if total >= n_samples:
                    break

                x     = images[i].unsqueeze(0).to(device)  # [1, C, H, W]
                label = labels[i].unsqueeze(0).to(device)  # [1]

                if targeted:
                    target_class = (labels[i].item() + 1) % 10
                    y_target = torch.tensor([target_class], device=device)
                    x_adv    = attack_fn(x, y_target)
                else:
                    target_class = None
                    x_adv = attack_fn(x, label)

                with torch.no_grad():
                    pred_orig = model(x).argmax(dim=1).item()
                    pred_adv  = model(x_adv).argmax(dim=1).item()

                if targeted:
                    is_success = (pred_adv == target_class)
                else:
                    is_success = (pred_adv != labels[i].item())

                success += is_success
                total   += 1

                if len(success_cases) < n_vis:
                    x_vis     = denorm_fn(x.cpu()).squeeze(0)     if denorm_fn else x.cpu().squeeze(0)
                    x_adv_vis = denorm_fn(x_adv.cpu()).squeeze(0) if denorm_fn else x_adv.cpu().squeeze(0)
                    perturbation = (x_adv_vis - x_vis).abs() * 10
                    perturbation = perturbation.clamp(0, 1)
                    entry = (labels[i].item(), pred_orig, pred_adv, x_vis, x_adv_vis, perturbation)
                    if is_success:
                        success_cases.append(entry)

            if total >= n_samples:
                break

    else:
        # ── GPU: 배치 단위 처리 ───────────────────────────────────────────
        for images, labels in tqdm(loader, desc=f"  {attack_name}", leave=False):
            if total >= n_samples:
                break

            # 마지막 배치 truncation (n_samples 초과 방지)
            remaining = n_samples - total
            images = images[:remaining]
            labels = labels[:remaining]

            x = images.to(device)

            if targeted:
                y_target = ((labels + 1) % 10).to(device)
                x_adv    = attack_fn(x, y_target)
            else:
                x_adv = attack_fn(x, labels.to(device))

            with torch.no_grad():
                pred_orig_batch = model(x).argmax(dim=1)
                pred_adv_batch  = model(x_adv).argmax(dim=1)

            pred_orig_cpu = pred_orig_batch.cpu()
            pred_adv_cpu  = pred_adv_batch.cpu()

            if targeted:
                is_success_batch = (pred_adv_cpu == (labels + 1) % 10)
            else:
                is_success_batch = (pred_adv_cpu != labels)

            success += is_success_batch.sum().item()
            total   += images.size(0)

            if len(success_cases) < n_vis:
                for i in range(images.size(0)):
                    if len(success_cases) >= n_vis:
                        break
                    if is_success_batch[i].item():
                        x_vis     = denorm_fn(x[i:i+1].cpu()).squeeze(0)     if denorm_fn else x[i:i+1].cpu().squeeze(0)
                        x_adv_vis = denorm_fn(x_adv[i:i+1].cpu()).squeeze(0) if denorm_fn else x_adv[i:i+1].cpu().squeeze(0)
                        perturbation = (x_adv_vis - x_vis).abs() * 10
                        perturbation = perturbation.clamp(0, 1)
                        entry = (labels[i].item(), pred_orig_cpu[i].item(), pred_adv_cpu[i].item(),
                                 x_vis, x_adv_vis, perturbation)
                        success_cases.append(entry)

    cases = success_cases[:n_vis]

    # ── 한꺼번에 저장 ─────────────────────────────────────────────────────
    def to_numpy(t):
        t = t.numpy()
        return t.squeeze() if dataset_name == "mnist" else t.transpose(1, 2, 0)

    for idx, (true_label, pred_orig, pred_adv, x_vis, x_adv_vis, perturbation) in enumerate(cases):
        if dataset_name == "cifar10":
            orig_label_str = CIFAR10_CLASSES[true_label]
            orig_pred_str  = CIFAR10_CLASSES[pred_orig]
            adv_pred_str   = CIFAR10_CLASSES[pred_adv]
        else:
            orig_label_str = str(true_label)
            orig_pred_str  = str(pred_orig)
            adv_pred_str   = str(pred_adv)

        _, axes = plt.subplots(1, 3, figsize=(9, 3))

        axes[0].imshow(to_numpy(x_vis),        cmap="gray" if dataset_name == "mnist" else None)
        axes[0].set_title(f"Original\nTrue: {orig_label_str}\nPred: {orig_pred_str}")
        axes[0].axis("off")

        axes[1].imshow(to_numpy(x_adv_vis),    cmap="gray" if dataset_name == "mnist" else None)
        axes[1].set_title(f"Adversarial\nPred: {adv_pred_str}")
        axes[1].axis("off")

        axes[2].imshow(to_numpy(perturbation), cmap="gray" if dataset_name == "mnist" else None)
        axes[2].set_title("Perturbation (×10)")
        axes[2].axis("off")

        plt.tight_layout()
        fname = f"{RESULTS_DIR}/{dataset_name}_{attack_name}_{idx+1}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

    if cases:
        print(f"  시각화 저장 완료: {RESULTS_DIR}/{dataset_name}_{attack_name}_1~{len(cases)}.png")

    return success / total * 100