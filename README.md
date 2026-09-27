# NaturaleMouseEvent: AI Natural Human Mouse Trajectory Engine

An AI-powered system designed to generate biologically authentic, human-like mouse cursor trajectories and interactions using behavioral biometrics, kinematics modeling, and PyTorch.

## Directory Structure & Files

- **`train_model.py`**: Production-grade PyTorch training pipeline implementing:
  - `ConditioningEncoder`: Multi-layer perceptron encoding start/target positions, 4D momentum context, binary intent, and latent Gaussian noise ($z \in \mathbb{R}^{16}$) to produce initial hidden states $h_0$ and step-wise context conditioning.
  - `KinematicDecoder`: 2-layer GRU with continuous context injection, bounded kinematics head (tanh-scaled displacement with $\le 0.05$ max step and sigmoid-scaled timing), and 4-class discrete action classification.
  - `DistanceSmoothLoss`: Masked multi-objective loss combining Huber kinematics loss, target reach L1 penalty, jerk/smoothness regularizer (2nd-order differences), and Cross-Entropy action classification.
  - Scheduled Sampling & Drift Stabilization: Decaying teacher forcing ratio ($1.0 \to 0.2$), screen boundary clamping, and padded-step freezing to completely eliminate autoregressive divergence during test rollouts.
- **`human_mouse.py`**: Self-contained high-level Python API module providing the `HumanMouse` class:
  - Global Biomechanical Safety Filter (`apply_biomechanical_kinematic_filter`): Enforces hardware-realistic step intervals ($\Delta t \ge 7.0\text{ ms}$), absolute velocity ceiling ($v \le 2200\text{ px/s}$), and acceleration limits ($|a| \le 35,000\text{ px/s}^2$) via forward-backward dynamic velocity profiling.
  - Normalized & Dwell-Decayed Momentum Handoff (`compute_terminal_momentum`): Bounded physiological handoff ($\|\mathbf{v}\| \le 1800\text{ px/s}$), screen dimension normalization, and exponential dwell dissipation ($\exp(-\Delta t_{\text{dwell}} / 0.08)$).
  - Terminal Deceleration & Submovement Settlement: Responsive stopping conditions eliminating pre-transition dead idle zones, combined with Flash & Hogan (1985) quintic minimum-jerk easing to $0.00\text{ px/s}$ target rest.
  - Continuous 8–12 Hz Ornstein-Uhlenbeck (OU) physiological tremor with ballistic speed suppression.
  - High-precision execution loop with Windows 1ms multimedia timer (`timeBeginPeriod`) and hybrid sleep-spinwait pacing.
- **`execute_mouse_action.py`**: Command-line interface (CLI) to execute natural cursor movements, sequences, and dry-run simulations with reach error reporting.
- **`preprocess_all_sqlite.py`**: Robust, memory-efficient data engineering pipeline that sequentially processes all SQLite telemetry databases (`data/*.sqlite3`), filters non-human / orthogonal grid artifacts, removes time anomalies, and exports unified training datasets (`data/mouse_dataset_fixed_N256_full.npz`).
- **`prepare_dataset.py`**: Legacy preprocessor for individual session database conversions.
- **`demo_test.py`**: Interactive demo and testing suite offering interactive terminal selection, dry-run statistical benchmarking, live on-screen waypoint navigation with Kinematic Momentum Chaining, precision clicking tests, idle wandering simulation, and kinematic trajectory visualization (`trajectory_demo.png`).
- **`simulate_mouse.py`**: Offline visualizer and simulation script for evaluating model trajectories.
- **`analyze_dur.py`**: Statistical analysis script measuring step time deltas ($\Delta t$) and episode duration distributions.
- **`scan_dbs.py`**: Database scanner tool to inspect raw session event counts and sequence lengths across `.sqlite3` files.
- **`train_pilot.py`**: Pilot training pipeline with Extended Motion History Context (Kinematic Memory):
  - Integrates 8D inter-episode history context vector $[v_{x0}, v_{y0}, \Delta t_{\text{dwell}}, \Delta x_{\text{jump}}, \Delta y_{\text{jump}}, \text{click\_density}, \text{avg\_dt}, \|\mathbf{v}_0\|]$ to maintain biomechanical momentum continuity across sequential goals.
  - Multi-objective `KinematicBioLoss` combining Smooth L1 kinematics, endpoint terminal reach, temporal cadence, discrete bio-jerk & total variation regularizer, boundary acceleration constraints ($\mathcal{L}_{\text{boundary}} = \|\Delta p_0 - \mathbf{v}_{\text{init}}\Delta t_0\|^2 + 0.5\|\Delta p_1 - \Delta p_0\|^2$), and Action Cross-Entropy.
  - Initial 10-step smoothstep muscle recruitment ramp bounding acceleration strictly below $90,000\text{ px/s}^2$ from rest.
- **`models/`**: Directory containing trained PyTorch neural network checkpoints (`pilot_mouse_model.pth`) and its [README.md](file:///D:/git/NaturaleMouseEvent/models/README.md).
- **`data/`**: Data directory containing preprocessed trajectory datasets (`mouse_dataset_fixed_N256_full.npz`) and its [README.md](file:///D:/git/NaturaleMouseEvent/data/README.md).
- **`human_mouse_engine/`**: Standalone packaged engine module and CLI utilities with documentation.

## Training the Model

### Pilot Training with Kinematic Memory (Local Subset)
Run the pilot training pipeline with 8D motion history context and bio-jerk regularization:

```bash
python train_pilot.py --data data/mouse_dataset_fixed_N256_full.npz --subset 6000 --epochs 6 --batch_size 64 --save_path models/pilot_mouse_model.pth
```

### Full Model Training (Production Pipeline)
Run the training pipeline with default parameters:

```bash
python train_model.py --epochs 40 --batch_size 64 --lr 0.001
```

Or with custom parameters and GPU acceleration:

```bash
python train_model.py --data data/mouse_dataset_fixed_N256_full.npz --epochs 50 --batch_size 128 --hidden_dim 256 --noise_dim 16 --tf_start 1.0 --tf_end 0.2 --save_path best_model.pth
```

## Interactive Demo & Testing

Run the interactive test suite (defaults to `models/pilot_mouse_model.pth` with 8D Kinematic Momentum Chaining):

```bash
python demo_test.py
```

Or execute direct automated modes:

```bash
# 1. Benchmark with Kinematic Momentum Chaining (Simulation only)
python demo_test.py --mode dry-run --trials 5

# 2. Live cursor test across 4 screen waypoints with dynamic curvature
python demo_test.py --mode live

# 3. Specify custom model checkpoint
python demo_test.py --mode live --model models/pilot_mouse_model.pth

# 4. Precision navigation and click test
python demo_test.py --mode click

# 5. Natural idle wandering and micro-tremor test
python demo_test.py --mode wander

# 6. Generate and export multi-segment kinematic chained graph (trajectory_demo.png)
python demo_test.py --mode plot
```
