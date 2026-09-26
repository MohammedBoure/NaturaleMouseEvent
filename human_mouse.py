import time
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import random

# Enable UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import pyautogui
    pyautogui.PAUSE = 0.0
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("[HumanMouse] Notice: 'pyautogui' is not installed. Running in dry-run mode (simulation only).")

# =====================================================================
# PyTorch Neural Network Architecture Definitions
# =====================================================================

class ConditioningEncoder(nn.Module):
    def __init__(self, noise_dim=16, hidden_dim=256, cond_dim=64, num_layers=2):
        super(ConditioningEncoder, self).__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers

        in_dim = 2 + 2 + 4 + 1 + noise_dim  # 9 + noise_dim

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

    def forward(self, start_pos, target_pos, prev_context, intent, z_noise=None):
        B = start_pos.size(0)
        if z_noise is None:
            z_noise = torch.randn(B, self.noise_dim, device=start_pos.device, dtype=start_pos.dtype)

        x = torch.cat([start_pos, target_pos, prev_context, intent, z_noise], dim=-1)
        feat = self.mlp(x)

        h0 = self.to_h0(feat).view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        cond_embed = self.to_context(feat)

        return h0, cond_embed


class KinematicDecoder(nn.Module):
    def __init__(self, hidden_dim=256, cond_dim=64, num_layers=2, num_classes=4, max_step_delta=0.08, max_dt=5.0):
        super(KinematicDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_step_delta = max_step_delta
        self.max_dt = max_dt

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

    def forward_step(self, step_input, h):
        gru_out, h_next = self.gru(step_input, h)
        out_flat = gru_out.squeeze(1)

        raw_kin = self.kinematics_head(out_flat)

        dx = torch.tanh(raw_kin[:, 0:1]) * self.max_step_delta
        dy = torch.tanh(raw_kin[:, 1:2]) * self.max_step_delta
        dt = torch.sigmoid(raw_kin[:, 2:3]) * self.max_dt

        kin_pred = torch.cat([dx, dy, dt], dim=-1)
        action_logits = self.action_head(out_flat)

        return kin_pred, action_logits, h_next


class HumanMouseGenerator(nn.Module):
    def __init__(self, noise_dim=16, hidden_dim=256, cond_dim=64, seq_len=256, num_layers=2, num_classes=4, max_step_delta=0.08):
        super(HumanMouseGenerator, self).__init__()
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_step_delta = max_step_delta

        self.encoder = ConditioningEncoder(
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

    def forward(self, start_pos, target_pos, prev_context, intent, z_noise=None, teacher_forcing_ratio=0.0, true_seq=None, padding_masks=None):
        B = start_pos.size(0)
        h, cond_embed = self.encoder(start_pos, target_pos, prev_context, intent, z_noise)

        curr_pos = start_pos.clone()
        prev_kin = torch.zeros(B, 3, device=start_pos.device, dtype=start_pos.dtype)

        pred_kinematics = []
        pred_actions = []
        pred_traj = []

        for t in range(self.seq_len):
            rem = torch.clamp(target_pos - curr_pos, min=-1.0, max=1.0)
            step_feat = torch.cat([prev_kin, rem, cond_embed], dim=-1).unsqueeze(1)

            kin_step, action_logits, h = self.decoder.forward_step(step_feat, h)

            pred_kinematics.append(kin_step)
            pred_actions.append(action_logits)

            next_dx_dy = kin_step[:, 0:2]
            next_dt = kin_step[:, 2:3]

            curr_pos = torch.clamp(curr_pos + next_dx_dy, min=-0.1, max=1.1)
            prev_kin = torch.cat([next_dx_dy, next_dt], dim=-1)

            pred_traj.append(curr_pos)

        pred_kinematics = torch.stack(pred_kinematics, dim=1)
        pred_actions = torch.stack(pred_actions, dim=1)
        pred_traj = torch.stack(pred_traj, dim=1)

        return pred_kinematics, pred_actions, pred_traj


# =====================================================================
# Trajectory Simulation Engine
# =====================================================================

class HumanMouseSimulator:
    def __init__(self, model_path="human_mouse_model.pt", display_w=1920, display_h=1080):
        self.display_w = float(display_w)
        self.display_h = float(display_h)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HumanMouseGenerator(
            noise_dim=16,
            hidden_dim=256,
            cond_dim=64,
            seq_len=256,
            num_layers=2,
            num_classes=4,
            max_step_delta=0.08
        ).to(self.device)

        resolved_path = None
        candidates = [
            model_path,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path),
            "human_mouse_model.pt",
            "best_model.pth",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_mouse_model.pt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
        ]
        for p in candidates:
            if p and os.path.exists(p):
                resolved_path = p
                break

        if resolved_path:
            checkpoint = torch.load(resolved_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[HumanMouse] Loaded trained AI mouse model from: {resolved_path}")
        else:
            print(f"[HumanMouse] Warning: Model file {model_path} not found. Using untrained weights.")

        self.model.eval()

    def generate_trajectory(self, start_pos, target_pos, prev_context=None, intent=1.0, target_radius=5.0):
        """
        Generates realistic human mouse trajectory from start_pos to target_pos.
        
        Args:
            start_pos: Tuple (x, y) in screen coordinates.
            target_pos: Tuple (x, y) in screen coordinates.
            prev_context: Optional 4D context tuple (vx, vy, click_density, avg_dt).
            intent: 1.0 for targeted movement/click, 0.0 for idle wandering.
            target_radius: Acceptable distance threshold (in px) to consider target reached.
            
        Returns:
            List of dict steps: [{'x': int, 'y': int, 'dt_ms': float, 'type': str}]
        """
        start_x, start_y = float(start_pos[0]), float(start_pos[1])
        target_x, target_y = float(target_pos[0]), float(target_pos[1])

        # If already at destination within target tolerance, return stationary point
        init_dist = np.hypot(target_x - start_x, target_y - start_y)
        if init_dist <= target_radius and intent > 0.5:
            return [{
                "x": int(round(start_x)),
                "y": int(round(start_y)),
                "dt_ms": 0.0,
                "type": "move"
            }]

        sx_norm = start_x / self.display_w
        sy_norm = start_y / self.display_h
        tx_norm = target_x / self.display_w
        ty_norm = target_y / self.display_h

        start_t = torch.tensor([[sx_norm, sy_norm]], dtype=torch.float32, device=self.device)
        target_t = torch.tensor([[tx_norm, ty_norm]], dtype=torch.float32, device=self.device)
        intent_t = torch.tensor([[intent]], dtype=torch.float32, device=self.device)

        if prev_context is None:
            ctx_t = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device)
        else:
            ctx_t = torch.tensor([[prev_context[0], prev_context[1], prev_context[2], prev_context[3]]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred_kin, pred_actions, pred_traj = self.model(
                start_t, target_t, ctx_t, intent_t, teacher_forcing_ratio=0.0
            )

        pred_kin = pred_kin[0].cpu().numpy()
        pred_actions = pred_actions[0].cpu().numpy()
        pred_traj = pred_traj[0].cpu().numpy()

        trajectory = []
        trajectory.append({
            "x": int(round(start_x)),
            "y": int(round(start_y)),
            "dt_ms": 0.0,
            "type": "move"
        })

        min_dist = float('inf')
        min_step = 0

        # Step through model output
        for step in range(len(pred_traj)):
            raw_x = pred_traj[step, 0] * self.display_w
            raw_y = pred_traj[step, 1] * self.display_h

            if step < 3:
                alpha = (step + 1) / 3.0
                x_px = (1 - alpha) * start_x + alpha * raw_x
                y_px = (1 - alpha) * start_y + alpha * raw_y
            else:
                x_px = raw_x
                y_px = raw_y

            # Clamp coordinates to physical screen dimensions
            x_px = max(0.0, min(self.display_w - 1.0, x_px))
            y_px = max(0.0, min(self.display_h - 1.0, y_px))

            dt_scaled = pred_kin[step, 2] * 100.0
            dt_ms = max(4.0, min(float(dt_scaled), 40.0))

            action_idx = int(np.argmax(pred_actions[step]))
            action_type = "move"
            if action_idx == 1:
                action_type = "click_down"
            elif action_idx == 2:
                action_type = "click_up"
            elif action_idx == 3:
                action_type = "scroll"

            trajectory.append({
                "x": int(round(x_px)),
                "y": int(round(y_px)),
                "dt_ms": round(dt_ms, 2),
                "type": action_type
            })

            dist_to_target = np.hypot(target_x - x_px, target_y - y_px)
            if dist_to_target < min_dist:
                min_dist = dist_to_target
                min_step = step

            # Natural stopping condition 1: reached destination within target tolerance
            if intent > 0.5 and dist_to_target <= target_radius and step > 10:
                break

            # Natural stopping condition 2: overshoot detection
            # If the model reached close proximity (<35px) and has started moving away
            # for multiple steps, truncate at the closest approach point.
            if intent > 0.5 and step > min_step + 8 and min_dist < 35.0 and dist_to_target > min_dist + 4.0:
                trajectory = trajectory[:min_step + 2]
                break

        # Natural Biological Convergence (Easing Micro-Correction):
        # If targeted movement ends slightly off-target (> target_radius), smoothly
        # converge using an exponential decay easing curve with microscopic neuromotor jitter.
        # NEVER perform a 1-step teleportation snap!
        if intent > 0.5:
            curr_x = float(trajectory[-1]["x"])
            curr_y = float(trajectory[-1]["y"])
            rem_dist = np.hypot(target_x - curr_x, target_y - curr_y)

            if rem_dist > target_radius:
                max_micro_steps = 35
                for _ in range(max_micro_steps):
                    rem_x = target_x - curr_x
                    rem_y = target_y - curr_y
                    d = np.hypot(rem_x, rem_y)
                    if d <= 1.2:
                        break

                    # Biological exponential decay easing
                    alpha = 0.32
                    jitter_x = float(np.random.normal(0, 0.15))
                    jitter_y = float(np.random.normal(0, 0.15))

                    dx = alpha * rem_x + jitter_x
                    dy = alpha * rem_y + jitter_y
                    step_mag = np.hypot(dx, dy)

                    # Bounded single micro-step displacement (<= 3.5 px) ensures smooth motion
                    max_step_len = 3.5
                    if step_mag > max_step_len:
                        dx = (dx / step_mag) * max_step_len
                        dy = (dy / step_mag) * max_step_len

                    curr_x += dx
                    curr_y += dy
                    curr_x = max(0.0, min(self.display_w - 1.0, curr_x))
                    curr_y = max(0.0, min(self.display_h - 1.0, curr_y))

                    dt_ms = float(np.random.uniform(6.5, 9.5))
                    trajectory.append({
                        "x": int(round(curr_x)),
                        "y": int(round(curr_y)),
                        "dt_ms": round(dt_ms, 2),
                        "type": "move"
                    })

        return trajectory

# =====================================================================
# High-Level HumanMouse Developer API
# =====================================================================

class HumanMouse:
    """
    High-level Programmatic API for natural AI human mouse movement.
    Uses trained PyTorch neural network weights to generate realistic human mouse trajectories.
    """
    def __init__(self, model_path=None, failsafe=True, dry_run=False):
        self.dry_run = dry_run
        if model_path is None:
            # First check current working directory, then script directory
            if os.path.exists("human_mouse_model.pt"):
                model_path = os.path.abspath("human_mouse_model.pt")
            elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_mouse_model.pt")):
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_mouse_model.pt")
            else:
                model_path = "human_mouse_model.pt"

        if PYAUTOGUI_AVAILABLE:
            self.screen_w, self.screen_h = pyautogui.size()
            pyautogui.FAILSAFE = failsafe
        else:
            self.screen_w, self.screen_h = 1920, 1080

        self.simulator = HumanMouseSimulator(
            model_path=model_path,
            display_w=self.screen_w,
            display_h=self.screen_h
        )
        self.prev_context = None

    def get_current_position(self):
        """Returns current mouse (x, y) position on screen."""
        if PYAUTOGUI_AVAILABLE:
            return pyautogui.position()
        return (100, 100)

    def generate_trajectory(self, start_pos, target_pos, prev_context=None, intent=1.0, target_radius=5.0):
        """Generates AI trajectory sequence without executing physical mouse movement."""
        return self.simulator.generate_trajectory(
            start_pos=start_pos,
            target_pos=target_pos,
            prev_context=prev_context if prev_context is not None else self.prev_context,
            intent=intent,
            target_radius=target_radius
        )

    def _execute_trajectory(self, trajectory, perform_click=False, button='left'):
        """Executes generated trajectory points with high-precision timing."""
        if not trajectory or self.dry_run:
            return

        if PYAUTOGUI_AVAILABLE:
            start_time = time.perf_counter()
            cumulative_target_sec = 0.0

            for step in trajectory:
                dt_sec = step['dt_ms'] / 1000.0
                cumulative_target_sec += dt_sec

                while True:
                    elapsed = time.perf_counter() - start_time
                    remaining = cumulative_target_sec - elapsed
                    if remaining <= 0:
                        break
                    if remaining > 0.002:
                        time.sleep(remaining - 0.001)

                x = step['x']
                y = step['y']
                try:
                    pyautogui.moveTo(x, y)
                except Exception:
                    pass

            if perform_click:
                try:
                    pyautogui.click(button=button)
                except Exception:
                    pass

    def _update_context(self, trajectory):
        """Calculates 4D velocity/momentum context for continuous smooth movements."""
        if len(trajectory) > 2:
            last_dx = (trajectory[-1]['x'] - trajectory[0]['x']) / self.screen_w
            last_dy = (trajectory[-1]['y'] - trajectory[0]['y']) / self.screen_h
            total_time_sec = sum(step['dt_ms'] for step in trajectory) / 1000.0
            clicks = sum(1 for step in trajectory if step['type'] in ['click_down', 'click_up'])
            avg_dt_ms = (total_time_sec * 1000.0) / len(trajectory)

            if total_time_sec > 0:
                self.prev_context = (
                    last_dx / total_time_sec,
                    last_dy / total_time_sec,
                    clicks / 256.0,
                    avg_dt_ms / 100.0
                )
            else:
                self.prev_context = (0.0, 0.0, 0.0, 0.0)

    def set_context(self, vx=0.0, vy=0.0, clicks=0.0, avg_dt=0.1):
        """
        Manually sets or overrides the 4D velocity/momentum context vector.
        Format: (vx, vy, clicks, avg_dt)
        """
        self.prev_context = (float(vx), float(vy), float(clicks), float(avg_dt))

    def get_context(self):
        """Returns current 4D context vector."""
        return self.prev_context

    def move_to(self, target_x, target_y, click=False, button='left', delay_after=0.1, prev_context=None, target_radius=5.0):
        """
        Smoothly moves mouse from current position to (target_x, target_y) naturally using AI.
        prev_context: Optional 4D tuple (vx, vy, clicks, avg_dt) to override velocity context.
        target_radius: Distance tolerance (px) to consider destination reached.
        """
        start_x, start_y = self.get_current_position()
        ctx = prev_context if prev_context is not None else self.prev_context
        trajectory = self.generate_trajectory((start_x, start_y), (target_x, target_y), prev_context=ctx, intent=1.0, target_radius=target_radius)

        self._execute_trajectory(trajectory, perform_click=click, button=button)
        self._update_context(trajectory)

        if delay_after > 0:
            time.sleep(delay_after)

        return trajectory

    def click_at(self, target_x, target_y, button='left', delay_after=0.1, prev_context=None, target_radius=5.0):
        """Moves mouse naturally to (target_x, target_y) and clicks."""
        return self.move_to(target_x, target_y, click=True, button=button, delay_after=delay_after, prev_context=prev_context, target_radius=target_radius)

    def wander(self, radius=200, delay_after=0.1):
        """Generates a natural idle wandering movement around the current area without intending to click."""
        start_x, start_y = self.get_current_position()
        target_x = max(0, min(self.screen_w, start_x + random.randint(-radius, radius)))
        target_y = max(0, min(self.screen_h, start_y + random.randint(-radius, radius)))
        
        trajectory = self.generate_trajectory((start_x, start_y), (target_x, target_y), prev_context=self.prev_context, intent=0.0)
        self._execute_trajectory(trajectory, perform_click=False)
        self._update_context(trajectory)

        if delay_after > 0:
            time.sleep(delay_after)

        return trajectory

    def move_sequence(self, target_list, click_targets=False, delay_between=0.4, target_radius=5.0):
        """Moves mouse smoothly across a sequence of targets (x, y) maintaining momentum context."""
        results = []
        for i, target in enumerate(target_list):
            target_x, target_y = target
            traj = self.move_to(
                target_x,
                target_y,
                click=click_targets,
                delay_after=delay_between if i < len(target_list) - 1 else 0.0,
                target_radius=target_radius
            )
            results.append(traj)
        return results

if __name__ == '__main__':
    print("Testing self-contained HumanMouse module dry-run...")
    mouse = HumanMouse()
    pos = mouse.get_current_position()
    print(f"Current Position: {pos}")
    traj = mouse.generate_trajectory(pos, (500, 500))
    print(f"Generated trajectory length: {len(traj)} steps")
