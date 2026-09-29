# NaturaleMouseEvent: AI Natural Human Mouse Trajectory Engine

An AI-powered system designed to generate biologically authentic, human-like mouse cursor trajectories and interactions using behavioral biometrics, kinematics modeling, and PyTorch.

## Directory Structure & Files

- **`train_model.py`**: Production-grade PyTorch training pipeline implementing:
  - `ConditioningEncoder`: Multi-layer perceptron encoding start/target positions, 4D momentum context, binary intent, and latent Gaussian noise ($z \in \mathbb{R}^{16}$) to produce initial hidden states $h_0$ and step-wise context conditioning.
  - `KinematicDecoder`: 2-layer GRU with continuous context injection, bounded kinematics head (tanh-scaled displacement with $\le 0.05$ max step and sigmoid-scaled timing), and 4-class discrete action classification.
  - `DistanceSmoothLoss`: Masked multi-objective loss combining Huber kinematics loss, target reach L1 penalty, jerk/smoothness regularizer (2nd-order differences), and Cross-Entropy action classification.
  - Scheduled Sampling & Drift Stabilization: Decaying teacher forcing ratio ($1.0 \to 0.2$), screen boundary clamping, and padded-step freezing to completely eliminate autoregressive divergence during test rollouts.
- **`human_mouse.py`**: Self-contained high-level Python API module providing the `HumanMouse` class:
  - Fitts's Law Dynamic Step Allocation: Distance-adaptive step budgeting ($N_{\text{steps}} = \text{int}(N_{\text{base}} + k \cdot \log_2(1 + \text{dist} / W_{\text{ref}}))$) allowing short reaches (200–300 px) to complete faster (~250–350 ms, 35–45 steps) and long reaches (800–1200 px) to scale naturally (~550–750 ms, 65–85 steps).
  - Natural Landing Phase & Box Dispersion: Decouples artificial 0.00 px robotic snapping, allowing authentic human terminal dispersion ($\sigma \approx 0.8\text{–}1.6\text{ px}$, average error ~1.5–2.5 px). Includes `sample_target_within_box(box_bounds)` for UI target selection via truncated 2D Gaussian ($\sigma = w/6, \sigma = h/6$).
  - Minimum-Jerk Transition Smoothing: At waypoint momentum handoffs, blends transition inflow momentum over the first 4–8 steps using a quintic minimum-jerk polynomial ($S(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$), eliminating discontinuous $\Delta a$ jerk spikes across chained waypoints.
  - Hardware Polling Micro-Jitter (Live Execution Layer): Replaces rigid sleep timing in `_execute_trajectory` with stochastic USB HID micro-jitter ($\Delta t \sim \mathcal{N}(8.0\text{ ms}, 0.6\text{ ms})$ clamped to $[4.0, 12.0]\text{ ms}$) emulating real physical 125 Hz USB mouse polling and OS thread scheduling quanta.
  - Global Biomechanical Safety Filter (`apply_biomechanical_kinematic_filter`): Enforces continuous Savitzky-Golay coordinate smoothing ($w=9, p=2$) with strict endpoint anchoring, distance-adaptive dynamic Fitts velocity ceiling ($v_{\text{ceiling}} = \max(4500, 160\sqrt{D})$) with continuous algebraic soft velocity saturation ($v_{\text{sat}} = v_{\text{raw}} / (1 + (v_{\text{raw}}/v_{\text{max}})^4)^{1/4}$), hardware polling bounds ($\Delta t \ge 7.0\text{ ms}$), and forward-backward acceleration profiling ($|a| \le 45,000\text{ px/s}^2$).
  - Directional Momentum Gating: Prevents acute retrograde hooks when incoming velocity opposes target direction (angle $\ge 90^\circ$).
  - Normalized & Dwell-Decayed Momentum Handoff (`compute_terminal_momentum`): Bounded physiological handoff ($\|\mathbf{v}\| \le 1800\text{ px/s}$), screen dimension normalization, and exponential dwell dissipation ($\exp(-\Delta t_{\text{dwell}} / 0.08)$).
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
- **`requirements.txt`**: Complete list of Python project dependencies across model training, preprocessing, filtering, visualization, and live automation.
- **`human_mouse_engine/`**: Standalone packaged engine module and CLI utilities with documentation.

## Training the Model

### Pilot Training with Kinematic Memory (Local Subset)
Run the pilot training pipeline with 8D motion history context and bio-jerk regularization on a local subset:

```bash
python train_pilot.py --data data/mouse_dataset_fixed_N256_full.npz --subset 6000 --epochs 6 --batch_size 64 --save_path models/pilot_mouse_model.pth
```

### Production Training (Full 53,028 Episodes on GPU / Colab T4)
Train across 100% of the trajectory dataset with Kinematic Memory, Automatic Mixed Precision (`torch.amp` FP16 Tensor Cores), and pinned GPU memory:

```bash
# Production training on all 53,028 episodes (90/10 split: 47,725 train / 5,303 val)
# Auto-enables AMP (FP16), pin_memory=True, and in-memory direct indexing (num_workers=0)
python train_pilot.py --full --epochs 10 --batch_size 128 --num_workers 0 --save_path models/production_mouse_model.pth

# Resume / Warm-Start from an existing checkpoint (e.g. continuing from Epoch 7 up to Epoch 12)
python train_pilot.py --full --epochs 12 --batch_size 128 --num_workers 0 --resume models/production_mouse_model.pth --save_path models/production_mouse_model.pth

# Or train on a custom number of episodes (e.g., 25,000)
python train_pilot.py --max_episodes 25000 --epochs 8 --batch_size 128 --num_workers 0
```

### Full Legacy Model Training (Baseline Pipeline)
Run the baseline training pipeline with default parameters:

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
