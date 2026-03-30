"""
fgsm.py
-------
FGSM(Fast Gradient Sign Method) 공격 구현.

- fgsm_targeted  : 특정 클래스로 오분류 유도 (loss 최소화)
- fgsm_untargeted: 단순 오분류 유도 (loss 최대화)

참고: 입력 x는 정규화된 상태로 들어온다고 가정.
     clamp 범위도 정규화 공간 기준으로 처리.
"""

import torch
import torch.nn as nn


def fgsm_targeted(model: nn.Module, x: torch.Tensor, y_target: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    FGSM targeted 공격: 입력 x를 y_target 클래스로 오분류하도록 교란한다.
    loss를 최소화하는 방향으로 perturbation을 계산한다.
    """
    x_adv = x.clone().detach().requires_grad_(True)

    # requires_grad_(True) 상태에서 forward해야 x_adv.grad 계산 가능
    # torch.no_grad() 사용 금지
    output = model(x_adv)
    loss = nn.CrossEntropyLoss()(output, y_target.to(x.device))

    model.zero_grad()
    loss.backward()

    # loss 최소화 방향(-) → target 클래스 쪽으로 이동
    x_adv = x_adv - epsilon * x_adv.grad.sign()

    return x_adv.detach()

def fgsm_untargeted(model: nn.Module, x: torch.Tensor, y_true: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    FGSM untargeted 공격: 입력 x를 y_true 클래스에서 오분류하도록 교란한다.
    loss를 최대화하는 방향으로 perturbation을 계산한다.
    """
    x_adv = x.clone().detach().requires_grad_(True)

    output = model(x_adv)
    loss = nn.CrossEntropyLoss()(output, y_true.to(x.device))

    model.zero_grad()
    loss.backward()

    # loss 최대화 방향(+) → 정답 클래스에서 멀어짐
    x_adv = x_adv + epsilon * x_adv.grad.sign()

    return x_adv.detach()