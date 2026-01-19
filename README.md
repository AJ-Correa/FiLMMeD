# FiLMMeD: Feature-wise Linear Modulation for Cross-Problem Multi-Depot Vehicle Routing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code for the paper:

> **FiLMMeD: Feature-wise Linear Modulation for Cross-Problem Multi-Depot Vehicle Routing**

## Overview

FiLMMeD is a unified neural solver for Multi-Task Learning (MTL) on Vehicle Routing Problems. We introduce three key innovations:

1. **FiLM Mechanism**: Feature-wise Linear Modulation to dynamically condition node embeddings based on active problem constraints
2. **Curriculum Learning**: A targeted training strategy for MDVRPs that progressively introduces more complex constraint combinations
3. **Preference Optimization**: Fine-tuning via PO instead of RL for single-depot VRP variants

FiLMMeD solves **24 MDVRP variants** (including 8 novel formulations with inter-depot routes) and **16 single-depot VRP variants**.

## Supported Problem Variants

| Constraint | Description |
|------------|-------------|
| **C** (Capacity) | Vehicle capacity limits |
| **O** (Open Route) | Vehicles don't return to depot |
| **B** (Backhaul) | Mixed pickup and delivery |
| **L** (Length Limit) | Route duration constraints |
| **TW** (Time Windows) | Service time windows |
| **I** (Inter-depot) | Vehicles can visit multiple depots |

## Repository Structure

```
FiLMMeD/
├── MTPOMO/          # FiLMMeD applied to MTPOMO architecture
├── MVMoE/           # FiLMMeD applied to MVMoE architecture  
├── CaDA/            # FiLMMeD applied to CaDA architecture (see separate README)
└── VRProblemDef.py  # Shared problem definitions
```

## Pre-trained Checkpoints

Each model includes trained checkpoints for:

| Checkpoint | Description |
|------------|-------------|
| `MDVRP50/` | Multi-depot VRP, 50 nodes (FiLM + Curriculum) |
| `MDVRP100/` | Multi-depot VRP, 100 nodes (FiLM + Curriculum) |
| `VRP50-PO/` | Single-depot VRP, 50 nodes (Preference Optimization) |
| `VRP100-PO/` | Single-depot VRP, 100 nodes (Preference Optimization) |

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/FiLMMeD.git
cd FiLMMeD

# Install dependencies
pip install torch torchvision torchaudio
pip install tensordict torchrl rich wandb pandas openpyxl PyYAML
```

## Training

### MTPOMO / MVMoE

```bash
cd MTPOMO  # or MVMoE

# Train on MDVRP (50 nodes)
python train.py --problem MDVRP --size 50 --film --curriculum

# Train on MDVRP (100 nodes)
python train.py --problem MDVRP --size 100 --film --curriculum
```

### Fine-tuning for Single-Depot VRPs

```bash
# Fine-tune using Preference Optimization
python tune.py --problem VRP --size 50 --po --checkpoint checkpoints/MDVRP50/checkpoint-300.pt
```

## Evaluation

```bash
# Test on MDVRP variants
python test.py --checkpoint checkpoints/MDVRP50/checkpoint-300.pt --size 50

# Test on benchmark instances
python test_benchmark.py --checkpoint checkpoints/MDVRP50/checkpoint-300.pt
```

## CaDA

CaDA has a different codebase structure. See [CaDA/README.md](CaDA/README.md) for specific instructions.

## Results

### MDVRP Variants (50 nodes)

| Method | Avg. Gap |
|--------|----------|
| MTPOMO | 7.605% |
| FiLMMeD-MTPOMO | **4.717%** |
| MVMoE | 6.587% |
| FiLMMeD-MVMoE | **4.740%** |

### Single-Depot VRPs (50 nodes)

| Method | Avg. Gap |
|--------|----------|
| MTPOMO | 4.602% |
| FiLMMeD-MTPOMO (PO) | **4.138%** |
| MVMoE | 4.352% |
| FiLMMeD-MVMoE (PO) | **4.081%** |

## Citation

If you find this work useful, please cite:

```bibtex
@article{correa2025filmed,
  title={FiLMMeD: Feature-wise Linear Modulation for Cross-Problem Multi-Depot Vehicle Routing},
  author={Corrêa, Arthur and Nascimento, Paulo and Moniz, Samuel},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MTPOMO](https://github.com/RoyalSkye/Routing-MVMoE) - Base architecture
- [MVMoE](https://github.com/RoyalSkye/Routing-MVMoE) - Mixture of Experts extension
- [CaDA](https://github.com/farkguidao/CaDA-MTVRP) - Dual-attention architecture
