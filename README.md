# adversarial-attacks

Implementation of FGSM and PGD adversarial attacks on MNIST and CIFAR-10 datasets using PyTorch.

---

## Project Structure

```
adversarial-attacks/
  src/
    datasets.py   : MNIST and CIFAR-10 data loading, normalization, denormalization
    models.py     : CNN for MNIST, ResNet18 for CIFAR-10
    train.py      : Model training module (can be run standalone)
    fgsm.py       : FGSM targeted/untargeted attack implementation
    pgd.py        : PGD targeted/untargeted attack implementation
    utils.py      : Attack success rate evaluation and visualization utilities
  results/        : Output directory for attack visualization PNGs
  test.py         : Training + full attack execution script
  requirements.txt
  report.pdf
```

## Models

| Dataset  | Architecture        | Test Accuracy |
|----------|---------------------|---------------|
| MNIST    | Custom CNN          | 98.38%        |
| CIFAR-10 | ResNet18            | 92.75%        |

ResNet18 architecture adapted from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) (MIT License).

## Attack Methods

| Attack           | Description                                                    |
|------------------|----------------------------------------------------------------|
| FGSM targeted    | Single-step perturbation in the loss-minimizing direction      |
| FGSM untargeted  | Single-step perturbation in the loss-maximizing direction      |
| PGD targeted     | Iterative FGSM targeted with eps-ball projection (k steps)     |
| PGD untargeted   | Iterative FGSM untargeted with eps-ball projection (k steps)   |

## Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run training + full attack evaluation (recommended)
```bash
python test.py
```
- If pretrained weights (`mnist_model.pth`, `cifar10_model.pth`) exist, skips training and runs attacks directly.
- Otherwise, trains the models automatically before running attacks.
- Results are saved as PNGs in `results/`.

### Run training only
```bash
python src/train.py
```

## Hyperparameters (`test.py`)

| Parameter             | Value              |
|-----------------------|--------------------|
| epsilon (eps)         | 0.05, 0.1, 0.2, 0.3 |
| PGD iterations (k)    | 40                 |
| PGD step size         | 0.01               |
| MNIST eval samples    | 1000               |
| CIFAR-10 eval samples | 100                |
| Visualization samples | 5 (success-first)  |

## References

- kuangliu/pytorch-cifar (MIT License) — https://github.com/kuangliu/pytorch-cifar
