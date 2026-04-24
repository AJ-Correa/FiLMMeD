# 🔬 FiLMMeD-CaDA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)](https://pytorch.org/)

FiLMMeD applied to the CaDA (Cross-Problem Routing Solver with Constraint-Aware Dual-Attention) architecture.

> Based on the original [CaDA](https://github.com/CIAM-Group/CaDA) by Li et al. (2025)

---

## 🎯 Overview

This implementation extends CaDA with FiLMMeD, specifically:

- ✨ **FiLM Mechanism**: Feature-wise Linear Modulation layer in the encoder
- 📈 **Preference Optimization**: Alternative to REINFORCE for improved generalization

---

## ⚙️ Installation

CaDA requires a specific environment, which we setup with Python 3.10 and the following package versions:

### Requirements

| Package | Version |
|:--------|:--------|
| Python | 3.10 |
| PyTorch | 2.0.1 |
| torchvision | 0.15.2 |
| torchaudio | 2.0.2 |
| tensordict | 0.1.2 |
| torchrl | 0.1.1 |
| rl4co | 0.2.0 |
| numpy | 1.24.4 |

### Setup Instructions

```bash
# Create and activate virtual environment with Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install PyTorch with CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies (order matters!)
pip install tensordict==0.1.2
pip install rl4co==0.2.0 --no-deps
pip install torchrl==0.1.1 --no-deps

# Install remaining packages
pip install numpy==1.24.4 pandas openpyxl PyYAML wandb einops rich
```

> ⚠️ **Note**: `rl4co` and `torchrl` must be installed with `--no-deps` to avoid version conflicts.

---

## 📁 Repository Structure

```
CaDA/
├── 📄 run.py           # Main entry point
├── 📄 config.yaml      # All hyperparameters
├── 📄 model.py         # VRPModel, Encoder, Decoder
├── 📄 trainer.py       # Training loop, testing, checkpointing
├── 📂 envs/            # Environment and instance generators
├── 📂 utils/           # Utility functions
├── 📂 data/            # Test datasets
└── 📂 checkpoints/     # Pre-trained models
```

---

## 🧠 Configuration (`config.yaml`)

All hyperparameters are configured in `config.yaml`. Command-line arguments override some settings.

### 🧠 Model Parameters (`model_params`)

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `embedding_dim` | Hidden dimension of node embeddings | `128` |
| `encoder_layer_num` | Number of transformer encoder layers | `6` |
| `qkv_dim` | Dimension of query/key/value vectors | `16` |
| `head_num` | Number of attention heads | `8` |
| `logit_clipping` | Clipping value for decoder logits | `10` |
| `ff_hidden_dim` | Feed-forward hidden dimension | `512` |
| `ffd` | Feed-forward activation function | `'siglu'` |
| `norm_type` | Normalization type (`'rms'` or `'layer'`) | `'rms'` |
| `use_sparse` | Sparse attention type (`'topk'`, `'relu'`, `'entmax15'`, etc.) | `'topk'` |
| `p_num` | Number of prompt vectors in PromptNet | `5` |
| `use_film` | ✨ Enable FiLM conditioning mechanism | `true` |

### 🎛️ Optimizer Parameters (`optimizer_params`)

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `optimizer.weight_decay` | Weight decay for AdamW | `1e-6` |
| `scheduler.name` | Learning rate scheduler | `'MultiStepLR'` |
| `scheduler.milestones` | Epochs to reduce LR | `[270, 295]` |
| `scheduler.gamma` | LR reduction factor | `0.1` |

> **Note**: Learning rate can be set via command line (`--lr`), default is `3e-4`.

### 🏋️ Trainer Parameters (`trainer_params`)

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `epochs` | Total training epochs | `300` |
| `train_step` | Training steps per epoch | `391` |
| `model_save_interval` | Save checkpoint every N epochs | `10` |
| `use_amp` | Enable automatic mixed precision | `true` |
| `loss_function` | `'po'` for Preference Optimization, `'rl'` for REINFORCE | `'po'` |
| `po_alpha` | PO entropy regularization coefficient | `0.03` |
| `po_B` | Solutions per instance in PO batch (`null` = default POMO size) | `null` |

### 🌍 Environment Parameters (`env`)

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `generator_params.num_loc` | Number of customer nodes (set via `--n_size`) | `50` |
| `generator_params.variant_preset` | Variant(s) to train on | `'all'` |
| `test_interval` | Run test every N epochs | `10` |
| `test_episodes` | Number of test instances | `1000` |
| `data_dir` | Directory for test data | `'./data/synthetic_data'` |

---

## 🚀 Command Line Arguments

| Argument | Description | Default |
|:---------|:------------|:--------|
| `--n_size` | Problem size (50 or 100) | `50` |
| `--batch_size` | Training batch size | `256` |
| `--lr` | Learning rate | `3e-4` |
| `--seed` | Random seed | `7` |
| `--test` | Enable testing during training | `false` |
| `--test_only` | Run testing only (no training) | `false` |
| `--resume` | Resume from checkpoint | `false` |
| `--epoch` | Checkpoint epoch to load | - |
| `--path_id` | Checkpoint folder path | - |
| `--wandb` | Weights & Biases project ID | `''` |

---

## 🏋️ Training

### Train on VRP with 50 nodes

Ensure `config.yaml` has:
```yaml
model_params:
  use_film: true

trainer_params:
  loss_function: 'po'
  po_alpha: 0.03
```

Then run:
```bash
python run.py --n_size 50 --batch_size 256 --test
```
> ⚠️ **Note**: `--test` ensures that, throughout training, the model is tested every `test_interval` epochs.

### Train on VRP with 100 nodes

```bash
python run.py --n_size 100 --batch_size 256 --test
```

---

## 🧪 Testing / Inference

### Test with pre-trained checkpoints

**VRP 50 nodes:**
```bash
python run.py --n_size 50 --test --test_only --resume --epoch 300 --path_id "../checkpoints/VRP50"
```

**VRP 100 nodes:**
```bash
python run.py --n_size 100 --test --test_only --resume --epoch 300 --path_id "../checkpoints/VRP100"
```

## 🏋️ Resume interrupted training

```bash
python run.py --n_size 50 --batch_size 256 --test --resume --epoch 150 --path_id "2025-0120-1230"
```
> ⚠️ **Note**: suppose training stopped at epoch 150, with the current checkpoint stored in `result/2025-0120-1230`.

---

## 📄 License

This project is licensed under the MIT License.