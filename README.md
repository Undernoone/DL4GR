# Curved CNN Demo

This repository contains a small experiment for comparing a normal CNN with a
CNN that uses a learnable local metric, inspired by the idea that representation
space can be curved instead of fixed and flat.

The demo is intentionally lightweight:

- `TraditionalCNN`: a simple convolutional encoder followed by a linear
  classifier.
- `CurvedMetricCNN`: the same encoder followed by a prototype classifier whose
  distances are measured with a sample-dependent metric `G(h)`.

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
python3 experiments/curved_cnn_demo.py --epochs 8 --seed 7
```

The script uses a synthetic image dataset, so it does not need internet access
or external data downloads.

Optional plot:

```bash
python3 experiments/curved_cnn_demo.py --epochs 8 --plot outputs/curved_cnn.png
```

All randomness is fixed by default: Python, Torch, CUDA/cuDNN deterministic
settings, synthetic sample generation, dataset split, and DataLoader shuffling
all use the same `--seed` value.
