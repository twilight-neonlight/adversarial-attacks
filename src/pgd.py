"""
pgd.py
------
PGD(Projected Gradient Descent) 공격 구현.

- pgd_targeted  : 특정 클래스로 오분류 유도 (loss 최소화, FGSM targeted을 k번 반복)
- pgd_untargeted: 단순 오분류 유도 (loss 최대화, FGSM untargeted을 k번 반복)

FGSM과의 차이:
- FGSM : 단일 스텝, 크기 eps
- PGD  : k번 반복, 스텝 크기 eps_step + 매 스텝마다 원본 x 기준 eps-ball로 projection
"""

import torch
import torch.nn as nn

def pgd_targeted(model: nn.Module, x: torch.Tensor, y_target: torch.Tensor,
                 k: int, eps: float, eps_step: float) -> torch.Tensor:
    """
    Targeted PGD 공격: 입력 x를 y_target 클래스로 오분류하도록 교란한다.
    loss를 최소화하는 방향으로 perturbation을 계산한다.
    """
    # 원본 x 저장 (매 스텝 projection 기준점으로 사용)
    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)

        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y_target.to(x.device))

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # loss 최소화 방향(-) → target 클래스 쪽으로 이동
            x_adv = x_adv - eps_step * x_adv.grad.sign()

            # eps-ball projection: delta를 [-eps, +eps]로 제한
            # clamp(0,1) 대신 원본 x 기준으로만 제한
            # (정규화된 입력은 [0,1] 범위를 벗어나므로 clamp(0,1) 사용 금지)
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = (x + delta).detach()

    return x_adv

def pgd_untargeted(model: nn.Module, x: torch.Tensor, y_true: torch.Tensor,
                   k: int, eps: float, eps_step: float) -> torch.Tensor:
    """
    Untargeted PGD 공격: 입력 x를 y_true 클래스에서 오분류하도록 교란한다.
    loss를 최대화하는 방향으로 perturbation을 계산한다.
    """
    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)

        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y_true.to(x.device))

        model.zero_grad()
        loss.backward()

        # loss 최대화 방향(+) → 정답 클래스에서 멀어짐
        with torch.no_grad():
            x_adv = x_adv + eps_step * x_adv.grad.sign()

            # 원본 x 기준 eps-ball로 projection
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = (x + delta).detach()

    return x_adv