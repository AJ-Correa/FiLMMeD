# 🚚 FiLMMeD: Feature-wise Linear Modulation for Cross-Problem Multi-Depot Vehicle Routing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official code for the paper:

> **FiLMMeD: Feature-wise Linear Modulation for Cross-Problem Multi-Depot Vehicle Routing**

---

## 🎯 Overview

FiLMMeD is a unified neural-based solver for Multi-Task Learning (MTL) on Vehicle Routing Problems. In total, FiLMMeD solves **24 MDVRP variants** (including 8 novel formulations with inter-depot routes) and **16 single-depot VRP variants**.

---

## 📁 Repository Structure

```
FiLMMeD/
├── 📂 MTPOMO/          # FiLMMeD applied to MTPOMO architecture
├── 📂 MVMoE/           # FiLMMeD applied to MVMoE architecture  
├── 📂 CaDA/            # FiLMMeD applied to CaDA architecture (see separate README)
└── 📄 VRProblemDef.py  # Problem instances generator
```

---

## 💾 Pre-trained Checkpoints

FiLMMeD-MTPOMO and FiLMMeD-MVMoE include trained checkpoints for:

| Checkpoint | Description |
|:-----------|:------------|
| `MDVRP50/` | Multi-depot VRP, 50 nodes |
| `MDVRP100/` | Multi-depot VRP, 100 nodes |
| `VRP50-PO/` | Single-depot VRP, 50 nodes |
| `VRP100-PO/` | Single-depot VRP, 100 nodes |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/AJ-Correa/FiLMMeD.git
cd FiLMMeD

# Install dependencies
pip install torch torchvision torchaudio
pip install tensordict torchrl rich wandb pandas openpyxl PyYAML
```

---

## 🔧 MTPOMO / MVMoE

Both models share the same parameter structure. The configuration is done by editing the parameter dictionaries directly in the Python files.

---

### 🏋️ Training (`train.py`)

To train the model, edit the parameters in `train.py` and run:

```bash
cd MTPOMO  # or MVMoE
python train.py
```

#### 🌍 Environment Parameters (`env_params`)

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `problem_type` | Variant(s) to train on. Use `'unified'` for single-constraint variants only, `'full_task_set'` for all variants, or specific variants like `'MDVRP'`, `'MDOVRPTW'`, etc. | `'full_task_set'` |
| `problem_size` | Number of customer nodes | `50` or `100` |
| `pomo_size` | Number of POMO starting nodes. Should be `problem_size + depot_size - 1` for MDVRP | `52` or `102` |
| `depot_size` | Number of depots | `3` |
| `curriculum` | Enable curriculum learning | `True` |
| `curriculum_schedule` | Epoch fractions to introduce 2-, 3-, and 4-constraint variants | `[0.3, 0.6, 0.9]` |

#### 🧠 Model Parameters (`model_params`)

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `embedding_dim` | Hidden dimension of embeddings | `128` |
| `encoder_layer_num` | Number of transformer encoder layers | `6` |
| `qkv_dim` | Dimension of query/key/value vectors | `16` |
| `head_num` | Number of attention heads | `8` |
| `logit_clipping` | Clipping value for logits | `10` |
| `ff_hidden_dim` | Feed-forward hidden dimension | `512` |
| `eval_type` | Decoding type during evaluation | `'argmax'` |
| `norm` | Normalization type | `'instance'` |
| `use_film` | ✨ Enable FiLM conditioning mechanism | `True` |

#### 🎛️ Trainer Parameters (`trainer_params`)

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `epochs` | Total training epochs | `300` |
| `train_episodes` | Training instances per epoch | `100000` |
| `train_batch_size` | Batch size | `128` for n=50, `64` for n=100 |
| `use_autocast` | Enable mixed precision training | `True` |
| `seed` | Random seed | `1234` |
| `model_save_interval` | Save checkpoint every N epochs | `20` |
| `model_load.enable` | Load from checkpoint to resume training | `False` |
| `model_load.path` | Path to checkpoint directory | `''` |
| `model_load.epoch` | Epoch to load from | `''` |

<details>
<summary>📋 <b>Settings to train FiLMMeD for MDVRP with 50 nodes</b> (click to expand)</summary>

Set the following in `train.py`:
```python
env_params = {
    'problem_type': 'full_task_set',
    'problem_size': 50,
    'pomo_size': 52,
    'depot_size': 3,
    'curriculum': True,
    'curriculum_schedule': [0.3, 0.6, 0.9]
}
model_params['use_film'] = True
```

Then run:
```bash
cd MTPOMO  # or MVMoE
python train.py
```
</details>

<details>
<summary>📋 <b>Settings to train FiLMMeD for MDVRP with 100 nodes</b> (click to expand)</summary>

Set the following in `train.py`:
```python
env_params = {
    'problem_type': 'full_task_set',
    'problem_size': 100,
    'pomo_size': 102,
    'depot_size': 3,
    'curriculum': True,
    'curriculum_schedule': [0.3, 0.6, 0.9]
}
model_params['use_film'] = True
```

Then run:
```bash
cd MTPOMO  # or MVMoE
python train.py
```
</details>

---

### 🎯 Fine-tuning for Single-Depot VRPs (`tune.py`)

After training on MDVRP, you can fine-tune the model on single-depot VRPs using Preference Optimization, as we did in our experiments.

#### 🌍 Environment Parameters for Fine-tuning

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `problem_type` | Same as MDVRP | `'full_task_set'` |
| `problem_size` | Number of customer nodes | `50` |
| `pomo_size` | Number of POMO starting nodes | `50` |
| `depot_size` | Always `1` for single-depot | `1` |

#### 🎛️ Tuner Parameters (`tuner_params`)

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `finetuning_epochs` | Number of fine-tuning epochs | `300` |
| `finetuning_episodes` | Instances per epoch | `100000` |
| `finetuning_batch_size` | Batch size | `128` for n=50, `64` for n=100 |
| `loss_type` | `'po_loss'` for Preference Optimization, `'rl_loss'` for REINFORCE | `'po_loss'` |
| `finetuning_lr` | Learning rate | `1e-4` |
| `alpha` | PO entropy regularization (only for `po_loss`) | `0.03` |
| `model_load.enable` | Set to `True` only if resuming an interrupted fine-tuning | `False` |
| `model_load.path` | Path to pre-trained MDVRP checkpoint | `''` |
| `model_load.epoch` | Epoch to load from | `''` |

<details>
<summary>📋 <b>Settings for VRP with 50 nodes</b> (from MDVRP50 checkpoint) (click to expand)</summary>

Set the following in `tune.py`:
```python
env_params = {
    'problem_type': 'full_task_set',
    'problem_size': 50,
    'pomo_size': 50,
    'depot_size': 1,
}
model_params['use_film'] = True
tuner_params['loss_type'] = 'po_loss'
tuner_params['model_load'] = {
    'enable': True,  # True only if resuming an interrupted fine-tuning
    'path': './checkpoints/MDVRP50/',  # Path to pre-trained MDVRP model
    'epoch': 300
}
```

Then run:
```bash
cd MTPOMO  # or MVMoE
python tune.py
```
</details>

<details>
<summary>📋 <b>Settings for VRP with 100 nodes</b> (from MDVRP100 checkpoint) (click to expand)</summary>

Set the following in `tune.py`:
```python
env_params = {
    'problem_type': 'full_task_set',
    'problem_size': 100,
    'pomo_size': 100,
    'depot_size': 1,
}
model_params['use_film'] = True
tuner_params['loss_type'] = 'po_loss'
tuner_params['model_load'] = {
    'enable': False,
    'path': './checkpoints/MDVRP100/',
    'epoch': 300
}
```

Then run:
```bash
cd MTPOMO  # or MVMoE
python tune.py
```
</details>

---

### 🧪 Testing / Inference (`test.py`)

To evaluate a trained model on test instances:

```bash
cd MTPOMO  # or MVMoE
python test.py
```

#### 🌍 Environment Parameters for Testing

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `problem_type` | `'full_task_set'` for all variants, or `'unified'` for single-constraint ones | `'full_task_set'` |
| `problem_size` | Number of customer nodes | `50` |
| `pomo_size` | Number of POMO starting nodes | `52` for MDVRP50, `102` for MDVRP100 |
| `depot_size` | Number of depots | `3` for MDVRP, `1` for single-depot |

#### 🎛️ Tester Parameters (`tester_params`)

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `model_load.path` | Path to checkpoint directory | `'./checkpoints/MDVRP50/'` |
| `model_load.epoch` | Epoch to load | `300` |
| `random_problems` | Generate random test instances | `False` |
| `num_random_problems` | Number of random test instances, if `random_problems is True` | `1000` |
| `test_batch_size` | Batch size for testing | `250` |
| `augmentation_type` | `'1'` (none), `'8'` (POMO x8), `'d'` (7+d), `'8d'` (8*d) | `'8d'` |
| `aug_batch_size` | Batch size when using augmentation | `250` |
| `enable_finetuning` | Enable on-the-fly fine-tuning during test | `False` |
| `use_autocast` | Mixed precision inference | `True` |

<details>
<summary>📋 <b>Settings for Testing MDVRP 50 nodes</b> (click to expand)</summary>

Set the following in `test.py`:
```python
env_params = {
    'problem_type': 'full_task_set',
    'problem_size': 50,
    'pomo_size': 52,
    'depot_size': 3,
}
model_params['use_film'] = True
tester_params['model_load'] = {
    'path': './checkpoints/MDVRP50/',
    'epoch': 300
}
tester_params['augmentation_type'] = '8d'
tester_params['random_problems'] = False
```

Then run:
```bash
cd MTPOMO  # or MVMoE
python test.py
```
</details>

<details>
<summary>📋 <b>Settings for Testing VRP 50 nodes</b> (single-depot) (click to expand)</summary>

Set the following in `test.py`:
```python
env_params = {
    'problem_type': 'full_task_set',
    'problem_size': 50,
    'pomo_size': 50,
    'depot_size': 1,
}
model_params['use_film'] = True
tester_params['model_load'] = {
    'path': './checkpoints/VRP50-PO/',
    'epoch': 300
}
tester_params['augmentation_type'] = '8'
tester_params['random_problems'] = False
```

Then run:
```bash
cd MTPOMO  # or MVMoE
python test.py
```
</details>

---

## 🔬 CaDA

CaDA has a different codebase structure. See [CaDA/README.md](CaDA/README.md) for specific instructions.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{correa2025filmed,
  title={FiLMMeD: Feature-wise Linear Modulation for Cross-Problem Multi-Depot Vehicle Routing},
  author={Corrêa, Arthur and Nascimento, Paulo and Moniz, Samuel},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [MTPOMO](https://github.com/FeiLiu36/MTNCO)
- [MVMoE](https://github.com/RoyalSkye/Routing-MVMoE)
- [CaDA](https://github.com/CIAM-Group/CaDA)
- [PO4COPs](https://github.com/MingjunPan/PO4COPs)
