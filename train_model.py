import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import argparse
import time

# Maximize multi-core CPU utilization (12 CPU cores)
num_cores = os.cpu_count() or 4
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditioningEncoder(nn.Module):
    def __init__(self, noise_dim=4, hidden_dim=256):
        super(ConditioningEncoder, self).__init__()
        # start_pos(2) + target_pos(2) + prev_context(4) + noise(dim) = 8 + noise_dim
        in_dim = 8 + noise_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, start_pos, target_pos, prev_context, noise):
        x = torch.cat([start_pos, target_pos, prev_context, noise], dim=-1)
        return self.fc(x)

class HumanMouseGenerator(nn.Module):
    def __init__(self, noise_dim=4, hidden_dim=256, seq_len=256, out_dim=4):
        super(HumanMouseGenerator, self).__init__()
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim

        self.encoder = ConditioningEncoder(noise_dim=noise_dim, hidden_dim=hidden_dim)
        self.gru = nn.GRU(input_size=out_dim + 4, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, start_pos, target_pos, prev_context, z_noise=None):
        batch_size = start_pos.size(0)
        if z_noise is None:
            z_noise = torch.randn(batch_size, self.noise_dim, device=start_pos.device)

        h0 = self.encoder(start_pos, target_pos, prev_context, z_noise)
        h = h0.unsqueeze(0).repeat(2, 1, 1)

        curr_pos = start_pos.clone()
        step_input = torch.zeros(batch_size, 1, 8, device=start_pos.device)
        step_input[:, 0, 4:6] = target_pos - start_pos
        step_input[:, 0, 6:8] = start_pos

        outputs = []
        cum_positions = []

        for t in range(self.seq_len):
            out_gru, h = self.gru(step_input, h)
            pred_step = self.head(out_gru.squeeze(1))
            outputs.append(pred_step)

            dx = pred_step[:, 0:1]
            dy = pred_step[:, 1:2]
            curr_pos = curr_pos + torch.cat([dx, dy], dim=-1)
            cum_positions.append(curr_pos)

            rem = target_pos - curr_pos
            step_input = torch.cat([pred_step, rem, curr_pos], dim=-1).unsqueeze(1)

        pred_seq = torch.stack(outputs, dim=1)
        pred_traj = torch.stack(cum_positions, dim=1)
        return pred_seq, pred_traj

class DistanceSmoothLoss(nn.Module):
    def __init__(self, w_target=10.0, w_path=2.0, w_smooth=1.0, w_speed=0.5, w_dt=2.0, w_action=1.0):
        super(DistanceSmoothLoss, self).__init__()
        self.w_target = w_target
        self.w_path = w_path
        self.w_smooth = w_smooth
        self.w_speed = w_speed
        self.w_dt = w_dt
        self.w_action = w_action
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_seq, pred_traj, true_seq, true_target, masks, start_pos):
        last_indices = masks.sum(dim=1).long() - 1
        last_indices = torch.clamp(last_indices, min=0, max=pred_traj.size(1) - 1)
        batch_indices = torch.arange(pred_traj.size(0), device=pred_traj.device)
        
        final_pred_pos = pred_traj[batch_indices, last_indices]
        l_target = torch.mean(torch.sum((final_pred_pos - true_target) ** 2, dim=-1))

        true_dx_dy = true_seq[:, :, 0:2]
        true_traj = start_pos.unsqueeze(1) + torch.cumsum(true_dx_dy, dim=1)
        
        path_diff = self.smooth_l1(pred_traj, true_traj).sum(dim=-1)
        l_path = (path_diff * masks).sum() / (masks.sum() + 1e-6)

        dx_dy = pred_seq[:, :, 0:2]
        accel = dx_dy[:, 1:, :] - dx_dy[:, :-1, :]
        jerk = accel[:, 1:, :] - accel[:, :-1, :]
        l_smooth = torch.mean(jerk ** 2)

        speeds = torch.norm(dx_dy, dim=-1)
        l_speed = torch.mean(torch.relu(speeds - 0.1) ** 2)

        # dt loss (index 2 in pred, index 2 in true)
        pred_dt = pred_seq[:, :, 2]
        true_dt = true_seq[:, :, 2]
        dt_diff = self.smooth_l1(pred_dt, true_dt)
        l_dt = (dt_diff * masks).sum() / (masks.sum() + 1e-6)

        # action loss (index 3 in pred, index 5 in true)
        pred_action = pred_seq[:, :, 3]
        true_action = true_seq[:, :, 5]
        action_diff = self.smooth_l1(pred_action, true_action)
        l_action = (action_diff * masks).sum() / (masks.sum() + 1e-6)

        total_loss = (self.w_target * l_target + 
                      self.w_path * l_path + 
                      self.w_smooth * l_smooth + 
                      self.w_speed * l_speed +
                      self.w_dt * l_dt +
                      self.w_action * l_action)

        return total_loss, {
            "l_target": l_target.item(),
            "l_path": l_path.item(),
            "l_smooth": l_smooth.item(),
            "l_speed": l_speed.item(),
            "l_dt": l_dt.item(),
            "l_action": l_action.item()
        }

def train():
    parser = argparse.ArgumentParser(description="Train Human Mouse Trajectory Generator AI (High-Speed Multi-Threaded)")
    parser.add_argument("--data", type=str, default=r"c:\Users\moham\Desktop\ai\naturale_mouse_event\mouse_dataset_fixed_N256_full.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (Optimized to 128 for 12 CPU cores)")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--save_path", type=str, default=r"c:\Users\moham\Desktop\ai\naturale_mouse_event\human_mouse_model.pt")
    args = parser.parse_args()

    print(f"Loading preprocessed dataset from: {args.data}")
    data = np.load(args.data)

    start_pos = torch.tensor(data['start_positions'], dtype=torch.float32)
    target_pos = torch.tensor(data['target_positions'], dtype=torch.float32)
    prev_ctx = torch.tensor(data['previous_contexts'], dtype=torch.float32)
    seq_tensors = torch.tensor(data['seq_tensors'], dtype=torch.float32)
    masks = torch.tensor(data['padding_masks'], dtype=torch.float32)

    dataset_size = start_pos.size(0)
    print(f"Dataset Loaded: {dataset_size} trajectory samples.")

    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset = torch.utils.data.TensorDataset(
        start_pos[:train_size], target_pos[:train_size], prev_ctx[:train_size],
        seq_tensors[:train_size], masks[:train_size]
    )
    val_dataset = torch.utils.data.TensorDataset(
        start_pos[train_size:], target_pos[train_size:], prev_ctx[train_size:],
        seq_tensors[train_size:], masks[train_size:]
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = HumanMouseGenerator(seq_len=256, hidden_dim=256).to(device)
    criterion = DistanceSmoothLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\n=======================================================")
    print("      STARTING FULL HARDWARE ACCELERATED TRAINING      ")
    print("=======================================================")
    print(f"Device: {device} | Allocated CPU Cores: {num_cores} / {os.cpu_count()}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}\n")

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        num_batches = len(train_loader)
        
        for batch_idx, (b_start, b_target, b_ctx, b_seq, b_mask) in enumerate(train_loader):
            b_start = b_start.to(device)
            b_target = b_target.to(device)
            b_ctx = b_ctx.to(device)
            b_seq = b_seq.to(device)
            b_mask = b_mask.to(device)

            optimizer.zero_grad()
            pred_seq, pred_traj = model(b_start, b_target, b_ctx)
            loss, loss_dict = criterion(pred_seq, pred_traj, b_seq, b_target, b_mask, b_start)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * b_start.size(0)
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - t0
                print(f"   [Epoch {epoch}] Batch {batch_idx + 1}/{num_batches} - Elapsed: {elapsed:.1f}s - Cur Loss: {loss.item():.4f}", end='\r')

        print() # Newline after epoch progress bar
        scheduler.step()
        train_loss /= train_size

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_target_dist = 0.0
        with torch.no_grad():
            for b_start, b_target, b_ctx, b_seq, b_mask in val_loader:
                b_start = b_start.to(device)
                b_target = b_target.to(device)
                b_ctx = b_ctx.to(device)
                b_seq = b_seq.to(device)
                b_mask = b_mask.to(device)

                pred_seq, pred_traj = model(b_start, b_target, b_ctx)
                loss, loss_dict = criterion(pred_seq, pred_traj, b_seq, b_target, b_mask, b_start)
                val_loss += loss.item() * b_start.size(0)

                final_pos = pred_traj[:, -1, :]
                dist_px = torch.norm((final_pos - b_target) * torch.tensor([1920.0, 1080.0], device=device), dim=-1)
                val_target_dist += dist_px.sum().item()

        val_loss /= val_size
        val_target_dist /= val_size
        dt_epoch = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epochs': epoch,
                'val_loss': val_loss,
                'val_target_dist_px': val_target_dist
            }, args.save_path)

        # Print EVERY epoch clearly with timing & metric feedback
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt_epoch:.2f}s) - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Avg Target Error: {val_target_dist:.1f} px")

    print("\n=======================================================")
    print("             TRAINING COMPLETED SUCCESSFULLY           ")
    print("=======================================================")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model Weights Saved To: {args.save_path}")
    print("=======================================================")

if __name__ == '__main__':
    train()
