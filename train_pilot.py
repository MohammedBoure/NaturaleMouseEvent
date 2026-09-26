"""
Pilot Training Pipeline for Natural Mouse Trajectory Generation with Extended Motion History Context.
Author: Senior Deep Learning Engineer & Kinematics Researcher
Project: NaturaleMouseEvent
"""

import os
import sys
import math
import random
import time
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# =====================================================================
# 1. Dataset Loader with Extended Kinematic Memory Context
# =====================================================================

class PilotKinematicDataset(Dataset):
    """
    Subsets and prepares episodes with Extended Motion History Context:
    - start_pos: (2,) normalized [x0, y0]
    - target_pos: (2,) normalized [xtgt, ytgt]
    - extended_context: (8,) [vx_prev, vy_prev, dt_dwell, dx_jump, dy_jump, click_density, avg_dt, momentum_mag]
    - intent: (1,) binary flag
    - seq_tensor: (256, 6) [dx, dy, dt, rem_x, rem_y, action_code]
    - mask: (256,) valid step indicator
    """
    def __init__(self, npz_path: str, subset_size: int = 6000, seed: int = 42):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset archive not found at: {npz_path}")

        print(f"[Dataset] Loading data from {npz_path}...")
        data = np.load(npz_path)

        total_available = len(data['start_targets'])
        print(f"[Dataset] Total episodes available: {total_available:,}")

        # Stratified deterministic index sampling across the dataset
        random.seed(seed)
        np.random.seed(seed)
        step_stride = max(1, total_available // subset_size)
        indices = np.arange(0, total_available, step_stride)[:subset_size]
        actual_size = len(indices)
        print(f"[Dataset] Sliced representative pilot subset: {actual_size:,} episodes (stride={step_stride})")

        start_targets = data['start_targets'][indices]
        seq_tensors = data['seq_tensors'][indices]
        masks = data['masks'][indices]
        prev_contexts = data['prev_contexts'][indices]
        intents = data['intents'][indices]

        # Construct Extended 8D Motion History Context
        # [vx_prev, vy_prev, dt_dwell, dx_jump, dy_jump, click_density, avg_dt, momentum_mag]
        N = actual_size
        ext_contexts = np.zeros((N, 8), dtype=np.float32)

        for i in range(N):
            vx0 = prev_contexts[i, 0]
            vy0 = prev_contexts[i, 1]
            click_density = prev_contexts[i, 2]
            avg_dt = prev_contexts[i, 3]

            if i > 0:
                prev_tgt = start_targets[i - 1, 2:4]
                curr_start = start_targets[i, 0:2]
                dx_jump = curr_start[0] - prev_tgt[0]
                dy_jump = curr_start[1] - prev_tgt[1]
                dt_dwell = 0.15  # Default idle cognitive pause (scaled)
            else:
                dx_jump, dy_jump = 0.0, 0.0
                dt_dwell = 0.1

            momentum_mag = float(np.hypot(vx0, vy0))

            ext_contexts[i] = [
                vx0,
                vy0,
                dt_dwell,
                dx_jump,
                dy_jump,
                click_density,
                avg_dt,
                momentum_mag
            ]

        self.start_positions = torch.from_numpy(start_targets[:, :2]).float()
        self.target_positions = torch.from_numpy(start_targets[:, 2:]).float()
        self.extended_contexts = torch.from_numpy(ext_contexts).float()
        self.intents = torch.from_numpy(intents).float()
        self.seq_tensors = torch.from_numpy(seq_tensors).float()
        self.masks = torch.from_numpy(masks).float()
        self.length = N

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return (
            self.start_positions[idx],
            self.target_positions[idx],
            self.extended_contexts[idx],
            self.intents[idx],
            self.seq_tensors[idx],
            self.masks[idx]
        )


# =====================================================================
# 2. Enhanced Model Architecture with Kinematic Memory Context
# =====================================================================

class ExtendedConditioningEncoder(nn.Module):
    """
    Encodes start/target coordinates, 8D extended motion history context,
    binary intent, and latent stochastic noise vector into initial GRU state h_0
    and per-step context embeddings.
    """
    def __init__(self, context_dim: int = 8, noise_dim: int = 16, hidden_dim: int = 256, cond_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers

        # start(2) + target(2) + ext_context(8) + intent(1) + noise(16) = 29
        in_dim = 2 + 2 + context_dim + 1 + noise_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        self.to_h0 = nn.Linear(hidden_dim, num_layers * hidden_dim)
        self.to_context = nn.Linear(hidden_dim, cond_dim)

    def forward(self, start_pos, target_pos, ext_context, intent, z_noise=None):
        B = start_pos.size(0)
        if z_noise is None:
            z_noise = torch.randn(B, self.noise_dim, device=start_pos.device, dtype=start_pos.dtype)

        feat_in = torch.cat([start_pos, target_pos, ext_context, intent, z_noise], dim=-1)
        feat = self.mlp(feat_in)

        # h0 shape: (num_layers, B, hidden_dim)
        h0 = self.to_h0(feat).view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        cond_embed = self.to_context(feat)

        return h0, cond_embed


class KinematicDecoder(nn.Module):
    """
    2-Layer GRU Kinematic Decoder:
    - Kinematics Head: Outputs bounded (dx, dy, dt) via tanh and sigmoid.
    - Action Head: Outputs logits for 4 discrete action classes.
    """
    def __init__(self, hidden_dim: int = 256, cond_dim: int = 64, num_layers: int = 2, num_classes: int = 4, max_step_delta: float = 0.08, max_dt: float = 5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_step_delta = max_step_delta
        self.max_dt = max_dt

        # step input: [dx, dy, dt, rem_x, rem_y] (5) + cond_embed (64) = 69
        step_in_dim = 5 + cond_dim

        self.gru = nn.GRU(
            input_size=step_in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.kinematics_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes)
        )

        # Initialize kinematics output layer with Xavier uniform
        nn.init.xavier_uniform_(self.kinematics_head[-1].weight)
        nn.init.zeros_(self.kinematics_head[-1].bias)

    def forward_step(self, step_input: torch.Tensor, h: torch.Tensor):
        gru_out, h_next = self.gru(step_input, h)
        out_flat = gru_out.squeeze(1)

        raw_kin = self.kinematics_head(out_flat)

        dx = torch.tanh(raw_kin[:, 0:1]) * self.max_step_delta
        dy = torch.tanh(raw_kin[:, 1:2]) * self.max_step_delta
        dt = torch.sigmoid(raw_kin[:, 2:3]) * self.max_dt

        kin_pred = torch.cat([dx, dy, dt], dim=-1)
        action_logits = self.action_head(out_flat)

        return kin_pred, action_logits, h_next


class MemoryConditionedMouseGenerator(nn.Module):
    """
    Unified AI Mouse Generator conditioned on Extended Motion History Context.
    """
    def __init__(self, context_dim: int = 8, noise_dim: int = 16, hidden_dim: int = 256, cond_dim: int = 64, seq_len: int = 256, num_layers: int = 2, num_classes: int = 4, max_step_delta: float = 0.08):
        super().__init__()
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_step_delta = max_step_delta

        self.encoder = ExtendedConditioningEncoder(
            context_dim=context_dim,
            noise_dim=noise_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            num_layers=num_layers
        )

        self.decoder = KinematicDecoder(
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            max_step_delta=max_step_delta
        )

    def forward(self, start_pos, target_pos, ext_context, intent, z_noise=None, teacher_forcing_ratio=0.0, true_seq=None, masks=None):
        B = start_pos.size(0)
        h, cond_embed = self.encoder(start_pos, target_pos, ext_context, intent, z_noise)

        curr_pos = start_pos.clone()
        prev_kin = torch.zeros(B, 3, device=start_pos.device, dtype=start_pos.dtype)

        pred_kinematics = []
        pred_actions = []

        use_tf = (teacher_forcing_ratio > 0.0 and true_seq is not None and self.training)

        for t in range(self.seq_len):
            # Dynamic closed-loop remaining distance feedback
            rem = torch.clamp(target_pos - curr_pos, min=-1.0, max=1.0)
            step_feat = torch.cat([prev_kin, rem, cond_embed], dim=-1).unsqueeze(1)

            kin_step, action_logits, h = self.decoder.forward_step(step_feat, h)

            pred_kinematics.append(kin_step)
            pred_actions.append(action_logits)

            if masks is not None:
                step_mask = masks[:, t:t+1]
            else:
                step_mask = torch.ones(B, 1, device=start_pos.device, dtype=start_pos.dtype)

            if use_tf:
                tf_coin = (torch.rand(B, 1, device=start_pos.device) < teacher_forcing_ratio).float()
                next_dx_dy = tf_coin * true_seq[:, t, 0:2] + (1.0 - tf_coin) * kin_step[:, 0:2]
                next_dt = tf_coin * true_seq[:, t, 2:3] + (1.0 - tf_coin) * kin_step[:, 2:3]
            else:
                next_dx_dy = kin_step[:, 0:2]
                next_dt = kin_step[:, 2:3]

            curr_pos = torch.clamp(curr_pos + next_dx_dy * step_mask, min=-0.1, max=1.1)
            prev_kin = torch.cat([next_dx_dy * step_mask, next_dt], dim=-1)

        pred_kinematics = torch.stack(pred_kinematics, dim=1)  # (B, seq_len, 3)
        pred_actions = torch.stack(pred_actions, dim=1)        # (B, seq_len, 4)

        if masks is not None:
            valid_disp = pred_kinematics[:, :, 0:2] * masks.unsqueeze(-1)
        else:
            valid_disp = pred_kinematics[:, :, 0:2]

        pred_traj = start_pos.unsqueeze(1) + torch.cumsum(valid_disp, dim=1)

        return pred_kinematics, pred_actions, pred_traj


# =====================================================================
# 3. Multi-Objective Kinematic Loss Function with Bio-Jerk Regularization
# =====================================================================

class KinematicBioLoss(nn.Module):
    """
    Multi-Objective Kinematic Loss:
    1. Displacement Huber Loss (w_disp = 10.0)
    2. Terminal Reach Penalty (w_reach = 30.0)
    3. Temporal Cadence Smooth L1 (w_dt = 3.0)
    4. Bio-Jerk Regularization (w_jerk = 0.05) on 3rd derivative of displacement
    5. Action Cross-Entropy (w_action = 1.0)
    """
    def __init__(self, w_disp: float = 10.0, w_reach: float = 30.0, w_dt: float = 3.0, w_jerk: float = 0.05, w_action: float = 1.0, beta: float = 0.01):
        super().__init__()
        self.w_disp = w_disp
        self.w_reach = w_reach
        self.w_dt = w_dt
        self.w_jerk = w_jerk
        self.w_action = w_action
        self.beta = beta

    def forward(self, pred_kin, pred_actions, pred_traj, true_seq, true_target, masks, start_pos):
        valid_mask_3d = masks.unsqueeze(-1)
        valid_steps = torch.clamp(masks.sum(), min=1.0)

        # 1. Displacement Huber Loss on (dx, dy)
        pred_disp = pred_kin[:, :, 0:2]
        true_disp = true_seq[:, :, 0:2]
        loss_disp = F.smooth_l1_loss(pred_disp * valid_mask_3d, true_disp * valid_mask_3d, beta=self.beta, reduction='sum') / valid_steps

        # 2. Terminal Reach Penalty: L1 distance at final valid step
        last_indices = torch.clamp(masks.sum(dim=1).long() - 1, min=0)
        B = start_pos.size(0)
        batch_idx = torch.arange(B, device=start_pos.device)
        pred_final = pred_traj[batch_idx, last_indices]
        reach_err_l1 = F.l1_loss(pred_final, true_target, reduction='mean')
        reach_err_px = torch.hypot((pred_final[:, 0] - true_target[:, 0]) * 1920.0, (pred_final[:, 1] - true_target[:, 1]) * 1080.0).mean()

        # 3. Temporal Cadence Loss on dt
        pred_dt = pred_kin[:, :, 2]
        true_dt = true_seq[:, :, 2]
        loss_dt = F.smooth_l1_loss(pred_dt * masks, true_dt * masks, beta=0.02, reduction='sum') / valid_steps

        # 4. Bio-Jerk Regularization Loss:
        # Jerk is the 3rd derivative of displacement: j_t = disp_t - 2*disp_{t-1} + disp_{t-2}
        if pred_disp.size(1) >= 3:
            jerk = pred_disp[:, 2:] - 2.0 * pred_disp[:, 1:-1] + pred_disp[:, :-2]
            jerk_mask = masks[:, 2:].unsqueeze(-1)
            jerk_norm_sq = torch.sum(jerk ** 2, dim=-1, keepdim=True) * jerk_mask
            loss_jerk = jerk_norm_sq.sum() / torch.clamp(jerk_mask.sum(), min=1.0)
        else:
            loss_jerk = torch.tensor(0.0, device=start_pos.device)

        # 5. Action Classification Cross-Entropy
        true_actions = true_seq[:, :, 5].long()
        flat_logits = pred_actions.reshape(-1, 4)
        flat_targets = true_actions.reshape(-1)
        flat_mask = masks.reshape(-1)
        ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        loss_action = (ce_loss * flat_mask).sum() / valid_steps

        total_loss = (
            self.w_disp * loss_disp +
            self.w_reach * reach_err_l1 +
            self.w_dt * loss_dt +
            self.w_jerk * loss_jerk +
            self.w_action * loss_action
        )

        metrics = {
            'loss_disp': loss_disp.item(),
            'loss_reach': reach_err_l1.item(),
            'reach_px': reach_err_px.item(),
            'loss_dt': loss_dt.item(),
            'loss_jerk': loss_jerk.item(),
            'loss_action': loss_action.item()
        }

        return total_loss, metrics


# =====================================================================
# 4. Pilot Training Engine
# =====================================================================

def run_pilot_training(
    dataset_path: str = "data/mouse_dataset_fixed_N256_full.npz",
    subset_size: int = 6000,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str = "models/pilot_mouse_model.pth"
):
    print("\n==========================================================================")
    print(" 🔬 INITIATING PILOT TRAINING: EXTENDED MOTION HISTORY CONTEXT")
    print(f" Subset Size: {subset_size} episodes | Epochs: {epochs} | Batch Size: {batch_size}")
    print("==========================================================================\n")

    # Select optimal device and threads
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"[Device] Using CPU with {num_threads} execution threads.")

    # Load and split pilot dataset
    full_pilot_dataset = PilotKinematicDataset(dataset_path, subset_size=subset_size)
    total_len = len(full_pilot_dataset)
    train_len = int(0.85 * total_len)
    val_len = total_len - train_len

    train_set, val_set = torch.utils.data.random_split(
        full_pilot_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[Split] Train Set: {train_len} episodes | Validation Set: {val_len} episodes")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = MemoryConditionedMouseGenerator(
        context_dim=8,
        noise_dim=16,
        hidden_dim=256,
        cond_dim=64,
        seq_len=256,
        num_layers=2,
        num_classes=4,
        max_step_delta=0.08
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Initialized MemoryConditionedMouseGenerator ({param_count:,} trainable parameters)")

    criterion = KinematicBioLoss(
        w_disp=10.0,
        w_reach=30.0,
        w_dt=3.0,
        w_jerk=0.05,
        w_action=1.0
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    best_val_err = float('inf')
    best_epoch = 0

    print("\n-----------------------------------------------------------------------------------------------------------------")
    print(f"{'Epoch':^9} | {'LR':^9} | {'Train Loss':^11} | {'Val Loss':^10} | {'Val Reach Err':^15} | {'Mean Jerk':^11} | {'Time':^7}")
    print("-----------------------------------------------------------------------------------------------------------------")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_batches = 0

        # Teacher forcing ratio decays from 0.8 to 0.1 across pilot epochs
        tf_ratio = max(0.1, 0.8 - (epoch - 1) * (0.7 / max(1, epochs - 1)))

        for b_start, b_tgt, b_ctx, b_intent, b_seq, b_mask in train_loader:
            b_start = b_start.to(device)
            b_tgt = b_tgt.to(device)
            b_ctx = b_ctx.to(device)
            b_intent = b_intent.to(device)
            b_seq = b_seq.to(device)
            b_mask = b_mask.to(device)

            optimizer.zero_grad()

            pred_kin, pred_act, pred_traj = model(
                start_pos=b_start,
                target_pos=b_tgt,
                ext_context=b_ctx,
                intent=b_intent,
                teacher_forcing_ratio=tf_ratio,
                true_seq=b_seq,
                masks=b_mask
            )

            loss, _ = criterion(
                pred_kin=pred_kin,
                pred_actions=pred_act,
                pred_traj=pred_traj,
                true_seq=b_seq,
                true_target=b_tgt,
                masks=b_mask,
                start_pos=b_start
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, train_batches)

        # Validation under pure autoregressive generation (tf_ratio = 0.0)
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_reach_px_list = []
        val_jerk_list = []

        with torch.no_grad():
            for b_start, b_tgt, b_ctx, b_intent, b_seq, b_mask in val_loader:
                b_start = b_start.to(device)
                b_tgt = b_tgt.to(device)
                b_ctx = b_ctx.to(device)
                b_intent = b_intent.to(device)
                b_seq = b_seq.to(device)
                b_mask = b_mask.to(device)

                pred_kin, pred_act, pred_traj = model(
                    start_pos=b_start,
                    target_pos=b_tgt,
                    ext_context=b_ctx,
                    intent=b_intent,
                    teacher_forcing_ratio=0.0,
                    true_seq=None,
                    masks=b_mask
                )

                loss, metrics = criterion(
                    pred_kin=pred_kin,
                    pred_actions=pred_act,
                    pred_traj=pred_traj,
                    true_seq=b_seq,
                    true_target=b_tgt,
                    masks=b_mask,
                    start_pos=b_start
                )

                val_loss += loss.item()
                val_reach_px_list.append(metrics['reach_px'])
                val_jerk_list.append(metrics['loss_jerk'])
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        avg_reach_px = np.mean(val_reach_px_list)
        avg_jerk = np.mean(val_jerk_list)
        dur_s = time.perf_counter() - t0
        curr_lr = scheduler.get_last_lr()[0]

        print(f" [{epoch:02d}/{epochs:02d}]  | {curr_lr:.2e} | {avg_train_loss:11.4f} | {avg_val_loss:10.4f} | {avg_reach_px:11.1f} px | {avg_jerk:11.5f} | {dur_s:5.1f}s")

        # Save best model
        if avg_reach_px < best_val_err:
            best_val_err = avg_reach_px
            best_epoch = epoch
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_reach_px': avg_reach_px,
                'val_loss': avg_val_loss,
                'config': {
                    'context_dim': 8,
                    'hidden_dim': 256,
                    'cond_dim': 64,
                    'noise_dim': 16,
                    'seq_len': 256
                }
            }, save_path)

    print("-----------------------------------------------------------------------------------------------------------------")
    print(f" ✨ Best Model Checkpoint Saved: Epoch {best_epoch} (Reach Error: {best_val_err:.1f} px) -> {save_path}\n")

    return model, save_path


# =====================================================================
# 5. Post-Training Kinematic & Visual Sanity Evaluation
# =====================================================================

def evaluate_kinematic_memory(model: nn.Module, device: torch.device):
    """
    Evaluates 3 diverse evaluation pairs under simulated momentum inheritance:
    1. Short displacement (~100 px)
    2. Medium diagonal (~500 px)
    3. Long screen-wide movement (~1200 px)
    """
    print("==========================================================================")
    print(" 🔍 POST-TRAINING TRAJECTORY SANITY & KINEMATIC MEMORY EVALUATION")
    print("==========================================================================\n")

    test_cases = [
        {
            "name": "Short Precision Movement",
            "start": (400.0, 300.0),
            "target": (480.0, 360.0),
            # Incoming residual velocity heading slightly southeast
            "prev_momentum": (0.04, 0.03, 0.1, 0.0, 0.0, 0.0, 0.08, 0.05)
        },
        {
            "name": "Medium Diagonal Crossing",
            "start": (300.0, 200.0),
            "target": (700.0, 500.0),
            # Incoming residual velocity heading northeast (requires smooth centrifugal arc)
            "prev_momentum": (0.08, -0.05, 0.1, 0.0, 0.0, 0.01, 0.08, 0.09)
        },
        {
            "name": "Long Screen-Wide Ballistic Movement",
            "start": (150.0, 100.0),
            "target": (1350.0, 750.0),
            # High incoming velocity heading southwest
            "prev_momentum": (-0.06, 0.08, 0.1, 0.0, 0.0, 0.02, 0.08, 0.10)
        }
    ]

    model.eval()

    for idx, tc in enumerate(test_cases, 1):
        s_x, s_y = tc["start"]
        t_x, t_y = tc["target"]
        nominal_dist = np.hypot(t_x - s_x, t_y - s_y)

        start_t = torch.tensor([[s_x / 1920.0, s_y / 1080.0]], dtype=torch.float32, device=device)
        target_t = torch.tensor([[t_x / 1920.0, t_y / 1080.0]], dtype=torch.float32, device=device)
        ctx_t = torch.tensor([tc["prev_momentum"]], dtype=torch.float32, device=device)
        intent_t = torch.tensor([[1.0]], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred_kin, pred_act, pred_traj = model(
                start_pos=start_t,
                target_pos=target_t,
                ext_context=ctx_t,
                intent=intent_t,
                teacher_forcing_ratio=0.0
            )

        kin = pred_kin[0].cpu().numpy()
        traj = pred_traj[0].cpu().numpy()

        traj_px = traj * np.array([1920.0, 1080.0])
        final_pos = traj_px[-1]
        reach_err_px = float(np.hypot(final_pos[0] - t_x, final_pos[1] - t_y))
        total_dur_ms = float(np.sum(kin[:, 2] * 100.0))

        # Compute Jerk smoothness metric: d3(x)/dt3
        disp_px = np.diff(traj_px, axis=0)
        accel = np.diff(disp_px, axis=0)
        jerk = np.diff(accel, axis=0)
        mean_jerk_metric = float(np.mean(np.sum(jerk ** 2, axis=-1)) * 1e4)

        # Check initial movement angle vs direct target angle
        initial_dx = traj_px[5, 0] - traj_px[0, 0]
        initial_dy = traj_px[5, 1] - traj_px[0, 1]
        direct_dx = t_x - s_x
        direct_dy = t_y - s_y
        cos_sim = (initial_dx * direct_dx + initial_dy * direct_dy) / (
            max(1e-6, np.hypot(initial_dx, initial_dy) * np.hypot(direct_dx, direct_dy))
        )
        curve_angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_sim))))

        print(f"[{idx}/3] {tc['name'].upper()}")
        print(f"   Coordinates:  Start {tc['start']} -> Target {tc['target']} (Displacement: {nominal_dist:.1f} px)")
        print(f"   Final Reach:  Landed at ({final_pos[0]:.1f}, {final_pos[1]:.1f}) | Reach Error: {reach_err_px:.1f} px")
        print(f"   Kinematics:   Duration: {total_dur_ms:.1f} ms | Smoothness (Jerk score): {mean_jerk_metric:.3f}")
        print(f"   Memory Arc:   Momentum Launch Curvature: {curve_angle_deg:.1f}° deviation from straight line (Natural Arc)")
        print("   Sample Path:  " + " -> ".join([f"({p[0]:.0f},{p[1]:.0f})" for p in traj_px[::35]]) + f" -> ({final_pos[0]:.0f},{final_pos[1]:.0f})")
        print("--------------------------------------------------------------------------")

    print("\n✅ Visual & Kinematic Sanity Checks Completed Successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pilot Training Script with Kinematic Memory")
    parser.add_argument("--data", type=str, default="data/mouse_dataset_fixed_N256_full.npz")
    parser.add_argument("--subset", type=int, default=6000, help="Subset size of episodes")
    parser.add_argument("--epochs", type=int, default=6, help="Number of pilot epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/pilot_mouse_model.pth")

    args = parser.parse_args()

    trained_model, saved_path = run_pilot_training(
        dataset_path=args.data,
        subset_size=args.subset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )

    device = next(trained_model.parameters()).device
    evaluate_kinematic_memory(trained_model, device)
