# Newton–Kepler

This is the github repo for the paper ["From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers"](https://arxiv.org/pdf/2602.06923).

Summary: [Vafa et al.](https://arxiv.org/abs/2507.06952) showed that a transformer can predict planetary motion (e.g., Earth orbiting the Sun) remarkably well, even though its internal representations do not explicitly encode gravitational forces. Does this mean transformers are not valid world models? **No.** In fact, our new paper shows that sometimes the transformer is learning a *Keplerian world model*, rather than a *Newtonian world model*. Even more intriguingly, when we replace full-context attention with *local attention* (i.e., reduce the context length), the transformer begins to behave like a **Newtonian world model**. As the context length is varied, we observe a sharp and fascinating *phase transition* between these two types of world models.

For more discussion about the implications for world models, read this [blog](https://kindxiaoming.github.io/blog/2026/kepler-newton/).

<img width="1060" height="719" alt="intro-plot" src="https://github.com/user-attachments/assets/b3223bb1-3dc2-4744-9278-8acc0eb6da0c" />

## Overview

Experiments and analysis for probing inductive biases of transformers on physics-inspired tasks: **1D sine waves** and **2D Kepler orbits** (Newtonian gravity).

This folder contains:

- **Training scripts** for classification and regression transformers on discretized trajectories.
- **Analysis notebooks** that reproduce figures and analyze training dynamics, spatial representations, scaling, and block-size (context length) effects.
- **Dataset generation** for sine waves and Kepler orbits.

Notebooks expect either pretrained checkpoints (Fig 2) or results from the training scripts; each notebook describes its dependencies in the first cells.

## Structure

| Path | Description |
|------|-------------|
| **fig2_vafa_spatial_map/** | Spatial map analysis of the pretrained transformer by Vafa et al. Loads a pretrained checkpoint and analyzes representations. **Requires checkpoint** (see below). |
| **fig3a_1d_sine_wave_embedding_evolution.ipynb** | 1D sine wave: embedding evolution over training. |
| **fig3b_1d_sine_wave_scaling_law.ipynb** | 1D sine wave: scaling law analysis. Uses results from `sine.py`. |
| **fig4_kepler_noisy_context_learning.ipynb** | Kepler: noisy context learning. |
| **fig5_regression_classification_comparison.ipynb** | Regression vs classification (Kepler). Uses results from `kepler.py` and `kepler_cv.py`. |
| **fig6_block_size_kepler_newton.ipynb** | Block size (context length) experiments for Kepler/Newton. Uses results from `kepler_cv_blocksize.py`. |
| **generate_dataset.py** | Generate and save large sine wave trajectory dataset. |
| **generate_kepler_cv.py** | Generate Kepler orbit trajectories (continuous variables) for training. |
| **sine.py** | Train transformers on 1D sine wave data. Saves to `./results/sine/`. |
| **kepler.py** | Train classification transformers on Kepler data. Saves to `./results/kepler/`. |
| **kepler_cv.py** | Train regression transformers on Kepler (continuous) data. Saves to `./results/kepler_cv/`. |
| **kepler_cv_blocksize.py** | Train with varying block size (context length). Saves to `./results/kepler_cv_blocksize/`. |
| **model.py**, **model_cv.py** | Model definitions (shared by training scripts). |
| **model_mlp.py** | MLP-only model — no attention, 2-hidden-layer MLP per block (ablation). |
| **linear_probe.py** | Standalone linear probe — compute R² on random/untrained models. |
| **kepler_cv_blocksize_nlayer1.py** | `n_layer=1` variant of the blocksize training script. |

## Checkpoint (Fig 2)

The **fig2_vafa_spatial_map** analysis loads a pretrained checkpoint. If `./ckpt` or `./ckpt.pt` is missing, the notebook will raise an error and ask you to download it.

1. Download the pretrained checkpoint (e.g. `ckpt.pt`) from [inductivebiasprobe repo](https://github.com/keyonvafa/inductive-bias-probes) or their [Google Drive link](https://drive.google.com/drive/folders/1qfdwQE2LpeZ1shGyrbX5y4xwze-DLbCF).
2. Place the file in **fig2_vafa_spatial_map/** as `./ckpt.pt`.
3. Re-run the notebook.

Run the notebook from **fig2_vafa_spatial_map/** so that `./ckpt.pt` is in the current working directory.

## Quick start

**Small toy models you can quickly train on a CPU**
* `fig3a_*.ipynb` contains a 1D sine-wave example.
* `fig4_*.ipynb` contains a 2D Kepler example. 

## MLP-only model (ablation: no attention)

`model_mlp.py` is a variant of `model_cv.py` where every Transformer block is replaced by a pure MLP block — **no attention at all**.

| | `model_cv.py` (original) | `model_mlp.py` (new) |
|---|---|---|
| Per-block structure | `LN → Attention → MLP(1 hidden)` | `LN → MLP2(2 hidden)` |
| MLP hidden layers | 1 (4×n_embd) | 2 (4×n_embd each) |
| Activation | SiLU | SiLU |
| Config class | `GPTConfigCV` | `GPTConfigCV` (same interface) |

### Architecture

```
Input (x, y) → Linear Embed → + Position Embed
    → MLPBlock① (LN → MLP2) → MLPBlock② (LN → MLP2) → ...
    → LayerNorm → Output Head → Prediction
```

`MLP2` forward: `Linear(n_embd → 4×n_embd) → SiLU → Linear(4×n_embd → 4×n_embd) → SiLU → Linear(4×n_embd → n_embd)`

### Usage

Drop-in replacement for `model_cv` in training scripts — just change the import:

```python
# Original (with attention)
from model_cv import GPTConfigCV, GPTCV

# MLP-only (no attention)
from model_mlp import GPTConfigCV, GPTCV
```

When using `kepler_cv_blocksize.py` with `model_mlp`, the hook registration must handle `MLPBlock` having no `attn` attribute. A pre-patched version is available as `kepler_cv_blocksize_mlp.py` (see below).

---

## `n_layer=1` training variant

`kepler_cv_blocksize_nlayer1.py` — identical to `kepler_cv_blocksize.py`, except all `n_layer` defaults are set to **1** instead of 2. Use this to train 1-layer models for fair parameter-count comparison with the 2-layer baseline (~25K params each).

| File | `n_layer` default |
|---|---|
| `kepler_cv_blocksize.py` | 2 |
| `kepler_cv_blocksize_nlayer1.py` | **1** |

Works with both `model_cv` and `model_mlp` — change the import as described above.

---

## Standalone linear probe (`linear_probe.py`)

Compute linear probe R² scores on a model **without any training** — the model is randomly initialized, activations are extracted via forward hooks, and linear regression probes are fit on the activations.

```bash
# Attention model, 2 layers
python linear_probe.py --model_type model_cv --n_layer 2 --num_trajectories 200

# MLP-only model, 1 layer
python linear_probe.py --model_type model_mlp --n_layer 1 --num_trajectories 200
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `model_mlp` | `model_cv` or `model_mlp` |
| `--n_layer` | 1 | Number of blocks |
| `--n_embd` | 32 | Embedding dimension |
| `--block_size` | 100 | Context length |
| `--input_dim` | 2 | Input dimension (x, y) |
| `--num_trajectories` | 500 | Number of test trajectories |
| `--seed` | 42 | Random seed |

Outputs per-layer R² scores (both `r2_all` on all positions and `r2_last` on the final timestep) for force targets (|F|, Fx, Fy, r, ...) and geometry targets (a, b, e, LRL, ...).

**Note:** Random models can produce non-zero R² because the input coordinates already contain geometric/force information, and random projections preserve some of it. Always compare against a trained baseline.

---

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. The list includes: PyTorch (>=2.0), NumPy, SciPy, Matplotlib, scikit-learn, PyYAML, and optionally tqdm for progress bars. Figure notebooks may use extra plotting options; install any missing packages as prompted.
