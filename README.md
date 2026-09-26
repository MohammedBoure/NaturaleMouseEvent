# NaturaleMouseEvent: AI Natural Human Mouse Trajectory Engine

An AI-powered system designed to generate biologically authentic, human-like mouse cursor trajectories and interactions using behavioral biometrics, kinematics modeling, and PyTorch.

## Directory Structure & Files

- **`train_model.py`**: Production-grade PyTorch training pipeline implementing:
  - `ConditioningEncoder`: Multi-layer perceptron encoding start/target positions, 4D momentum context, binary intent, and latent Gaussian noise ($z \in \mathbb{R}^{16}$) to produce initial hidden states $h_0$ and step-wise context conditioning.
  - `KinematicDecoder`: 2-layer GRU with continuous context injection and dual-head output (Kinematics Head for continuous $(\Delta x, \Delta y, \Delta t)$ and Action Head for 4-class discrete mouse actions).
  - `DistanceSmoothLoss`: Masked multi-objective loss combining Huber kinematics loss, target reach L1 penalty, jerk/smoothness regularizer (2nd-order differences), and Cross-Entropy action classification.
  - Scheduled Sampling: Decaying teacher forcing ratio ($1.0 \to 0.2$) over epochs to prevent autoregressive drift during test rollouts.
- **`human_mouse.py`**: Self-contained high-level Python API module providing the `HumanMouse` class for generating and executing natural trajectories with sub-millisecond precision timing.
- **`execute_mouse_action.py`**: Command-line interface (CLI) to execute natural cursor movements, sequences, and dry-run simulations.
- **`prepare_dataset.py`**: Preprocessing script converting raw SQLite recording databases into padded, normalized `.npz` tensor datasets.
- **`simulate_mouse.py`**: Offline visualizer and simulation script for evaluating model trajectories.
- **`analyze_dur.py`**: Statistical analysis script measuring step time deltas ($\Delta t$) and episode duration distributions.
- **`scan_dbs.py`**: Database scanner tool to inspect raw session event counts and sequence lengths across `.sqlite3` files.
- **`data/`**: Data directory containing preprocessed trajectory datasets (`mouse_dataset_fixed_N256_full.npz`) and its [README.md](file:///D:/git/NaturaleMouseEvent/data/README.md).
- **`human_mouse_engine/`**: Standalone packaged engine module and CLI utilities with documentation.

## Training the Model

Run the training pipeline with default parameters:

```bash
python train_model.py --epochs 40 --batch_size 64 --lr 0.001
```

Or with custom parameters and GPU acceleration:

```bash
python train_model.py --data data/mouse_dataset_fixed_N256_full.npz --epochs 50 --batch_size 128 --hidden_dim 256 --noise_dim 16 --tf_start 1.0 --tf_end 0.2 --save_path best_model.pth
```
