"""
utils.py
--------
공격 결과 평가 및 시각화 유틸리티.

- run_attack: 공격 실행, 성공률 계산, 시각화 저장을 한 번의 루프에서 처리
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

    Returns:
        success_rate: 공격 성공률 (0.0 ~ 100.0)
    """
    model.eval()
    success       = 0
    total         = 0
    success_cases = []
    fail_cases    = []

    for images, labels in loader:
        for i in range(images.size(0)):
            # 성공률용 n_samples 및 시각화용 케이스 수집이 모두 끝나면 중단
            if total >= n_samples and \
               len(success_cases) >= n_vis and len(fail_cases) >= n_vis:
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

            # 성공 여부 판정
            if targeted:
                is_success = (pred_adv == target_class)
            else:
                is_success = (pred_adv != labels[i].item())

            # ── 성공률 카운트 (n_samples개까지) ───────────────────────────
            if total < n_samples:
                success += is_success
                total   += 1

            # ── 시각화용 케이스 수집 ───────────────────────────────────────
            if len(success_cases) < n_vis or len(fail_cases) < n_vis:
                x_vis     = denorm_fn(x.cpu()).squeeze(0)     if denorm_fn else x.cpu().squeeze(0)
                x_adv_vis = denorm_fn(x_adv.cpu()).squeeze(0) if denorm_fn else x_adv.cpu().squeeze(0)
                perturbation = (x_adv_vis - x_vis).abs() * 10
                perturbation = perturbation.clamp(0, 1)

                entry = (labels[i].item(), pred_orig, pred_adv, x_vis, x_adv_vis, perturbation)
                if is_success:
                    if len(success_cases) < n_vis:
                        success_cases.append(entry)
                else:
                    if len(fail_cases) < n_vis:
                        fail_cases.append(entry)

        if total >= n_samples and \
           len(success_cases) >= n_vis and len(fail_cases) >= n_vis:
            break

    # ── 성공 케이스 우선으로 n_vis개 채우기 ──────────────────────────────
    cases    = success_cases[:n_vis]
    shortage = n_vis - len(cases)
    if shortage > 0:
        print(f"  ※ 공격 성공 케이스 부족 ({len(success_cases)}개) → 실패 케이스로 보충")
        cases += fail_cases[:shortage]

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
