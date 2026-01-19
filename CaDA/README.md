# FiLMMeD-CaDA

This is the CaDA (Cross-Problem Routing Solver with Constraint-Aware Dual-Attention) implementation with FiLMMeD enhancements.

> Based on the original [CaDA](https://github.com/CIAM-Group/CaDA) by Li et al.

## FiLMMeD Enhancements

CaDA has been modified with the following FiLMMeD contributions:

1. **FiLM Mechanism**: Feature-wise Linear Modulation layer in the encoder (`model.py`)
2. **Curriculum Learning**: Progressive variant introduction during training (configurable in `config.yaml`)
3. **Preference Optimization**: Alternative loss function for improved convergence (`config.yaml`: `loss_function: 'po'`)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Key packages:
# torch >= 2.0.1
# torchrl >= 0.1.1
# tensordict >= 0.1.2
# rich, wandb, pandas, openpyxl
```

## Configuration

All settings are controlled via `config.yaml`:

```yaml
model_params:
  use_film: true           # Enable FiLM conditioning

trainer_params:
  epochs: 300
  loss_function: 'po'      # 'po' for Preference Optimization, 'rl' for REINFORCE
  po_alpha: 0.03           # PO entropy regularization
  use_amp: true            # Mixed precision training
```

## Training

### Single-GPU Training

```bash
python run.py --n_size 50 --batch_size 256 --seed 42
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=4 run.py --n_size 50 --batch_size 64 --ddp
```

### Resume from Checkpoint

```bash
python run.py --n_size 50 --resume --epoch 100 --path result/your_run/
```

## Evaluation

```bash
# Test with augmentation
python run.py --n_size 50 --test_only --resume --epoch 300 --path result/your_run/

# Skip evaluation during training
python run.py --n_size 50 --skip_test
```

## Key Files

| File | Description |
|------|-------------|
| `run.py` | Main entry point with DDP and optimization flags |
| `config.yaml` | All hyperparameters and settings |
| `model.py` | VRPModel, Encoder, Decoder, FiLM, PromptNet |
| `trainer.py` | Training loop, testing, checkpointing |
| `envs/env.py` | MTVRP environment |
| `envs/generator.py` | Instance generation with variant sampling |

## FiLM Architecture

The FiLM layer modulates node embeddings based on the constraint vector `[C, O, TW, L, B]`:

```python
class FiLM(nn.Module):
    def forward(self, x, cond):
        gamma = self.gamma(cond)  # Scale
        beta = self.beta(cond)    # Shift
        return gamma * x + beta
```

This allows the model to dynamically adapt its representations depending on which constraints are active.

## Data

Test datasets are located in `data/`. The generator creates instances on-the-fly during training.

## Results

FiLMMeD-CaDA achieves state-of-the-art performance on 16 single-depot VRP variants, outperforming prior MTL models including RouteFinder and the original CaDA.

## Citation

```bibtex
@article{correa2025filmed,
  title={FiLMMeD: Feature-wise Linear Modulation for Cross-Problem Multi-Depot Vehicle Routing},
  author={Corrêa, Arthur and Nascimento, Paulo and Moniz, Samuel},
  year={2025}
}

@article{li2024cada,
  title={CaDA: Cross-Problem Routing Solver with Constraint-Aware Dual-Attention},
  author={Li, Jiaming and others},
  journal={arXiv preprint arXiv:2412.00346},
  year={2024}
}
```
