# Trained Models Directory

This directory stores trained PyTorch neural network checkpoints, weights, and evaluation artifacts for the AI Natural Human Mouse system.

## Checkpoints

### 1. `pilot_mouse_model.pth`
- **Architecture**: `MemoryConditionedMouseGenerator`
  - `ExtendedConditioningEncoder`: Encodes 8D Extended Motion History Context (`[vx_prev, vy_prev, dt_dwell, dx_jump, dy_jump, click_density, avg_dt, momentum_mag]`), start/target screen positions, binary intent flag, and 16D Gaussian stochastic noise ($z \sim \mathcal{N}(0, I)$).
  - `KinematicDecoder`: 2-layer GRU with step-wise context projection injection, bounded continuous kinematics head ($\tanh$-scaled $\le 0.08$ step displacement, sigmoid-scaled $\Delta t$), and 4-class discrete action classification (`Move`, `Press`, `Release`, `Scroll`).
- **Total Parameters**: 901,767 trainable parameters.
- **Training Strategy**: Local pilot training run on 6,000 stratified episodes from `data/mouse_dataset_fixed_N256_full.npz` (5,100 train / 900 val) for 6 epochs with AdamW, Cosine Annealing, and multi-objective `KinematicBioLoss` (Displacement Smooth L1, Terminal Reach L1, Temporal Cadence L1, Discrete Bio-Jerk & Total Variation regularizer, Boundary Constraints $\mathcal{L}_{\text{boundary}} = \|\Delta p_0 - \mathbf{v}_{\text{init}}\Delta t_0\|^2 + 0.5\|\Delta p_1 - \Delta p_0\|^2$, and Action Cross-Entropy).
- **Biomechanical Kinematic Safeguards**:
  - *Acceleration Spike Elimination*: Initial displacement modulated by a 10-step smoothstep muscle recruitment ramp ($\tau^2(3-2\tau)$), strictly bounding initial acceleration below $90,000\text{ px/s}^2$ (typical starting $a_0 \approx 10,000\text{–}30,000\text{ px/s}^2$).
  - *Physiological Tremor Modeling*: Replaced per-step white Gaussian noise with a continuous 8–12 Hz Ornstein-Uhlenbeck (OU) process ($\theta=15.0, \sigma=0.22$) with ballistic velocity suppression.
  - *Tail Plateau Elimination*: Replaced constant-velocity clamping floors with Flash & Hogan (1985) quintic minimum-jerk polynomial deceleration down to a terminal $0.00\text{ px/s}$ resting state.
- **Validation Results**:
  - Validation Loss: $1.1755 \to 0.1245$
  - Validation Reach Error: $983.8\text{ px} \to 44.7\text{ px}$ (0.00 px reach error with minimum-jerk settlement)
- **Trajectory Sanity Benchmarks**:
  - *Short Precision Movement (100 px)*: Final Reach Error: 0.00 px, Natural Arc
  - *Medium Diagonal Crossing (500 px)*: Final Reach Error: 0.00 px, Natural Arc
  - *Long Ballistic Movement (1365 px)*: Final Reach Error: 0.00 px, Natural Arc
