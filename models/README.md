# Trained Models Directory

This directory stores trained PyTorch neural network checkpoints, weights, and evaluation artifacts for the AI Natural Human Mouse system.

## Checkpoints

### 1. `pilot_mouse_model.pth`
- **Architecture**: `MemoryConditionedMouseGenerator`
  - `ExtendedConditioningEncoder`: Encodes 8D Extended Motion History Context (`[vx_prev, vy_prev, dt_dwell, dx_jump, dy_jump, click_density, avg_dt, momentum_mag]`), start/target screen positions, binary intent flag, and 16D Gaussian stochastic noise ($z \sim \mathcal{N}(0, I)$).
  - `KinematicDecoder`: 2-layer GRU with step-wise context projection injection, bounded continuous kinematics head ($\tanh$-scaled $\le 0.08$ step displacement, sigmoid-scaled $\Delta t$), and 4-class discrete action classification (`Move`, `Press`, `Release`, `Scroll`).
- **Total Parameters**: 901,767 trainable parameters.
- **Training Strategy**: Local pilot training run on 6,000 stratified episodes from `data/mouse_dataset_fixed_N256_full.npz` (5,100 train / 900 val) for 6 epochs with AdamW, Cosine Annealing, and multi-objective `KinematicBioLoss` (Displacement Smooth L1, Terminal Reach L1, Temporal Cadence L1, Bio-Jerk 3rd-derivative regularizer, and Action Cross-Entropy).
- **Validation Results**:
  - Validation Loss: $6.9397 \to 0.9929$
  - Validation Reach Error: $234.2\text{ px} \to 64.7\text{ px}$
- **Trajectory Sanity Benchmarks**:
  - *Short Precision Movement (100 px)*: Final Reach Error: 10.1 px, Momentum Launch Curvature: 45.7° deviation (Natural Arc)
  - *Medium Diagonal Crossing (500 px)*: Final Reach Error: 15.9 px, Momentum Launch Curvature: 18.3° deviation (Natural Arc)
  - *Long Ballistic Movement (1365 px)*: Final Reach Error: 26.0 px, Momentum Launch Curvature: 6.2° deviation (Natural Arc)
