# Curved CNN Demo

This repository contains a small experiment for comparing a normal CNN with a
CNN that uses a learnable local metric, inspired by the idea that representation
space can be curved instead of fixed and flat. It supports public torchvision
datasets as well as a tiny synthetic smoke-test dataset.

The demo is intentionally lightweight:

- `TraditionalCNN`: a simple convolutional encoder followed by a linear
  classifier.
- `ResidualCNN`: a non-geometric residual CNN baseline with a structure closer
  to the geometric-flow model.
- `CurvedMetricCNN`: the same encoder followed by a prototype classifier whose
  distances are measured with a sample-dependent metric `G(h)`.
- `GeometricFlowCNN`: a CNN whose internal feature maps are updated by learned
  metric and curvature fields:

```text
X_{l+1} = G_theta(X_l, g_theta(X_l), kappa_theta(X_l))
```

In the curved model, the classifier logit for class `c` is:

```text
logit_c = -(h - p_c)^T G(h) (h - p_c) + b_c
```

For this first demo, `G(h)` is a positive diagonal matrix predicted from the
current feature vector. This is a practical first approximation of a local
Riemannian metric: the model can stretch or compress different feature
directions depending on the sample.

## Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the comparison:

```bash
python3 experiments/curved_cnn_demo.py --dataset mnist --epochs 8 --seed 7
```

Supported datasets:

- `mnist`
- `fashion-mnist`
- `cifar10`
- `synthetic`

Public datasets are downloaded to `data/` by default. Use `--no-download` if
the dataset already exists on the server and the run should not touch the
network.

Optional plot:

```bash
python3 experiments/curved_cnn_demo.py \
  --dataset cifar10 \
  --device cuda \
  --epochs 30 \
  --batch-size 512 \
  --models traditional,residual,geometric-flow \
  --scheduler cosine \
  --label-smoothing 0.05 \
  --seed 7 \
  --metrics-json outputs/cifar10_metrics.json \
  --plot outputs/cifar10_curved_cnn.png
```

Run only the geometric-flow model:

```bash
python3 experiments/curved_cnn_demo.py \
  --dataset cifar10 \
  --device cuda \
  --epochs 30 \
  --batch-size 512 \
  --models geometric-flow \
  --curvature-reg 0.0001 \
  --metric-reg 0.00001 \
  --scheduler cosine \
  --label-smoothing 0.05 \
  --seed 7 \
  --metrics-json outputs/cifar10_geometric_flow.json
```

By default, public training datasets use light augmentation. For CIFAR10 this
means random crop plus horizontal flip. Use `--no-augment` for an ablation.
Runs also use cosine learning-rate decay and label smoothing by default; use
`--scheduler none` or `--label-smoothing 0.0` for ablations.

Geometric-flow runs print extra diagnostics:

- `reg`: the added geometric regularization term.
- `g`: mean and standard deviation of the learned metric field.
- `|k|`: mean absolute curvature.
- `step`: learned residual flow step size.

All randomness is fixed by default: Python, Torch, CUDA/cuDNN deterministic
settings, synthetic sample generation, dataset split, and DataLoader shuffling
all use the same `--seed` value.
