import time
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import random

# Windows Real-Time Multimedia Timer & Kernel Cursor Subsystems
WINMM_AVAILABLE = False
USER32_AVAILABLE = False
winmm = None
user32 = None

if os.name == 'nt':
    import ctypes
    try:
        winmm = ctypes.WinDLL('winmm')
        WINMM_AVAILABLE = True
    except Exception:
        winmm = None
    try:
        user32 = ctypes.windll.user32
        USER32_AVAILABLE = True
    except Exception:
        user32 = None

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
# PyTorch Neural Network Architecture Definitions
# =====================================================================

class ExtendedConditioningEncoder(nn.Module):
    """
    Encodes start/target coordinates, motion history context (8D or 4D),
    binary intent, and latent stochastic noise vector into initial GRU state h_0
    and per-step context embeddings.
    """
    def __init__(self, context_dim=8, noise_dim=16, hidden_dim=256, cond_dim=64, num_layers=2):
        super(ExtendedConditioningEncoder, self).__init__()
        self.context_dim = context_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers

        # start(2) + target(2) + ext_context(context_dim) + intent(1) + noise(16)
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

        h0 = self.to_h0(feat).view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        cond_embed = self.to_context(feat)

        return h0, cond_embed


# Backward compatibility alias
ConditioningEncoder = ExtendedConditioningEncoder


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


class MemoryConditionedMouseGenerator(nn.Module):
    def __init__(self, context_dim=8, noise_dim=16, hidden_dim=256, cond_dim=64, seq_len=256, num_layers=2, num_classes=4, max_step_delta=0.08):
        super(MemoryConditionedMouseGenerator, self).__init__()
        self.context_dim = context_dim
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

    def forward(self, start_pos, target_pos, ext_context, intent, z_noise=None, teacher_forcing_ratio=0.0, true_seq=None, padding_masks=None):
        B = start_pos.size(0)
        h, cond_embed = self.encoder(start_pos, target_pos, ext_context, intent, z_noise)

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


# Backward compatibility alias
HumanMouseGenerator = MemoryConditionedMouseGenerator


# =====================================================================
# Trajectory Simulation Engine
# =====================================================================

class HumanMouseSimulator:
    def __init__(self, model_path=None, display_w=1920, display_h=1080):
        self.display_w = float(display_w)
        self.display_h = float(display_h)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_dim = 8

        # Dynamic model checkpoint discovery with priority order:
        # 1. User-supplied model_path
        # 2. models/pilot_mouse_model.pth (trained extended-context model)
        # 3. best_model.pth
        # 4. human_mouse_model.pt
        resolved_path = None
        script_dir = os.path.dirname(os.path.abspath(__file__))

        candidate_paths = []
        if model_path:
            candidate_paths.extend([
                model_path,
                os.path.join(script_dir, model_path)
            ])
        candidate_paths.extend([
            "models/pilot_mouse_model.pth",
            os.path.join(script_dir, "models/pilot_mouse_model.pth"),
            "best_model.pth",
            os.path.join(script_dir, "best_model.pth"),
            "human_mouse_model.pt",
            os.path.join(script_dir, "human_mouse_model.pt"),
            "human_mouse_engine/human_mouse_model.pt",
            os.path.join(script_dir, "human_mouse_engine/human_mouse_model.pt")
        ])

        for p in candidate_paths:
            if p and os.path.exists(p):
                resolved_path = os.path.abspath(p)
                break

        if resolved_path:
            checkpoint = torch.load(resolved_path, map_location=self.device, weights_only=False)

            # Auto-detect context dimension from checkpoint config or weight shapes
            detected_context_dim = 8
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                detected_context_dim = checkpoint['config'].get('context_dim', 8)
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
                if 'encoder.mlp.0.weight' in state:
                    detected_context_dim = state['encoder.mlp.0.weight'].shape[1] - 21
            elif isinstance(checkpoint, dict) and 'encoder.mlp.0.weight' in checkpoint:
                detected_context_dim = checkpoint['encoder.mlp.0.weight'].shape[1] - 21

            self.context_dim = max(4, detected_context_dim)
            self.model = MemoryConditionedMouseGenerator(
                context_dim=self.context_dim,
                noise_dim=16,
                hidden_dim=256,
                cond_dim=64,
                seq_len=256,
                num_layers=2,
                num_classes=4,
                max_step_delta=0.08
            ).to(self.device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            print(f"[HumanMouse] Loaded trained AI mouse model from: {resolved_path} (context_dim={self.context_dim})")
        else:
            self.context_dim = 8
            self.model = MemoryConditionedMouseGenerator(
                context_dim=8,
                noise_dim=16,
                hidden_dim=256,
                cond_dim=64,
                seq_len=256,
                num_layers=2,
                num_classes=4,
                max_step_delta=0.08
            ).to(self.device)
            print(f"[HumanMouse] Warning: Model file {model_path} not found. Using untrained weights.")

        self.model.eval()

    def generate_trajectory(self, start_pos, target_pos, prev_context=None, intent=1.0, target_radius=5.0):
        """
        Generates realistic human mouse trajectory from start_pos to target_pos
        conditioned on continuous kinematic memory context.
        
        Args:
            start_pos: Tuple (x, y) in screen coordinates.
            target_pos: Tuple (x, y) in screen coordinates.
            prev_context: Optional context tuple (8D or 4D) carrying residual momentum.
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

        # Context alignment across 8D and 4D representations
        if prev_context is None:
            if self.context_dim == 8:
                ctx_arr = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0]
            else:
                ctx_arr = [0.0, 0.0, 0.0, 0.1]
        elif len(prev_context) == 4 and self.context_dim == 8:
            vx0, vy0, clicks, avg_dt = prev_context
            mag = float(np.hypot(vx0, vy0))
            ctx_arr = [float(vx0), float(vy0), 0.1, 0.0, 0.0, float(clicks), float(avg_dt), mag]
        elif len(prev_context) == 8 and self.context_dim == 4:
            ctx_arr = [float(prev_context[0]), float(prev_context[1]), float(prev_context[5]), float(prev_context[6])]
        elif len(prev_context) == self.context_dim:
            ctx_arr = [float(c) for c in prev_context]
        else:
            ctx_arr = [float(c) for c in list(prev_context)[:self.context_dim]]

        ctx_t = torch.tensor([ctx_arr], dtype=torch.float32, device=self.device)

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

        # Neuromuscular Tremor Biomechanics:
        # Human motor execution contains 8-12 Hz physiological tremor (~0.25 - 0.40 px amplitude).
        # Modeled as an autoregressive Ornstein-Uhlenbeck / AR(1) colored noise process.
        tremor_x, tremor_y = 0.0, 0.0
        rho = 0.75  # Correlation factor for ~10-12 Hz oscillation at ~120 Hz sampling
        tremor_base_std = 0.35  # Biomechanical tremor standard deviation (px)

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

            dist_to_target = np.hypot(target_x - x_px, target_y - y_px)
            if dist_to_target < min_dist:
                min_dist = dist_to_target
                min_step = step

            # Smoothly damp tremor amplitude as cursor approaches target
            tremor_damping = min(1.0, max(0.05, dist_to_target / 30.0))
            eta_x = float(np.random.normal(0, tremor_base_std * tremor_damping))
            eta_y = float(np.random.normal(0, tremor_base_std * tremor_damping))
            tremor_x = rho * tremor_x + np.sqrt(1.0 - rho ** 2) * eta_x
            tremor_y = rho * tremor_y + np.sqrt(1.0 - rho ** 2) * eta_y

            final_x = max(0.0, min(self.display_w - 1.0, x_px + tremor_x))
            final_y = max(0.0, min(self.display_h - 1.0, y_px + tremor_y))

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
                "x": int(round(final_x)),
                "y": int(round(final_y)),
                "dt_ms": round(dt_ms, 2),
                "type": action_type
            })

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

        # Micro-Freeze & Stair-Stepping Elimination:
        # Merge consecutive identical integer coordinates to prevent zero-velocity stalls
        if len(trajectory) > 1:
            filtered_trajectory = [trajectory[0]]
            for s in trajectory[1:]:
                prev = filtered_trajectory[-1]
                if s["x"] == prev["x"] and s["y"] == prev["y"] and s["type"] == prev["type"]:
                    prev["dt_ms"] = round(prev["dt_ms"] + s["dt_ms"], 2)
                else:
                    filtered_trajectory.append(s)
            trajectory = filtered_trajectory

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
        self.failsafe = failsafe
        # Simulator will prioritize user model_path, then models/pilot_mouse_model.pth, best_model.pth, etc.

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
        """Executes generated trajectory points with sub-millisecond precision timing and Windows multimedia timer."""
        if not trajectory or self.dry_run:
            return

        # Boost Windows system timer resolution to 1ms
        timer_boosted = False
        if winmm is not None:
            try:
                winmm.timeBeginPeriod(1)
                timer_boosted = True
            except Exception:
                pass

        try:
            start_time = time.perf_counter()
            cumulative_target_sec = 0.0

            for step in trajectory:
                dt_sec = step['dt_ms'] / 1000.0
                cumulative_target_sec += dt_sec
                step_target_time = start_time + cumulative_target_sec

                # Sub-millisecond hybrid sleep-spinwait pacing
                while True:
                    rem = step_target_time - time.perf_counter()
                    if rem <= 0.0:
                        break
                    # If remaining time > 2ms, sleep coarsened by 1.2ms to avoid scheduler oversleep
                    if rem > 0.002:
                        time.sleep(rem - 0.0012)
                    # For final sub-millisecond fraction, spin-wait to eliminate quantum latency

                x = int(step['x'])
                y = int(step['y'])

                # Fail-safe check: escape to screen corner (0, 0) if not starting at (0, 0)
                if self.failsafe and (x, y) == (0, 0) and (trajectory[0]['x'], trajectory[0]['y']) != (0, 0):
                    raise pyautogui.FailSafeException("PyAutoGUI fail-safe triggered from mouse moving to corner (0, 0)")

                # Low-latency direct kernel cursor positioning on Windows, fallback to pyautogui
                if user32 is not None:
                    user32.SetCursorPos(x, y)
                elif PYAUTOGUI_AVAILABLE:
                    try:
                        pyautogui.moveTo(x, y)
                    except Exception:
                        pass

            if perform_click and PYAUTOGUI_AVAILABLE:
                try:
                    pyautogui.click(button=button)
                except Exception:
                    pass

        finally:
            # Restore system timer resolution
            if timer_boosted and winmm is not None:
                try:
                    winmm.timeEndPeriod(1)
                except Exception:
                    pass

    def compute_terminal_momentum(self, trajectory, dwell_time_sec=0.1, jump_offset=(0.0, 0.0)):
        """
        Computes the residual arrival velocity vector (vx, vy) and builds the full
        kinematic memory context vector for smooth momentum chaining across waypoints.
        """
        if not trajectory or len(trajectory) < 2:
            if getattr(self.simulator, 'context_dim', 8) == 8:
                return (0.0, 0.0, float(dwell_time_sec), 0.0, 0.0, 0.0, 0.1, 0.0)
            return (0.0, 0.0, 0.0, 0.1)

        # Look at the final 6 to 10 points before any micro-easing to capture arrival velocity
        k = min(10, len(trajectory) - 1)
        dx_px = float(trajectory[-1]['x'] - trajectory[-k]['x'])
        dy_px = float(trajectory[-1]['y'] - trajectory[-k]['y'])
        dt_sum_sec = float(sum(trajectory[j]['dt_ms'] for j in range(len(trajectory) - k, len(trajectory))) / 1000.0)

        if dt_sum_sec > 1e-4:
            vx_final = (dx_px / self.screen_w) / dt_sum_sec
            vy_final = (dy_px / self.screen_h) / dt_sum_sec
        else:
            vx_final, vy_final = 0.0, 0.0

        momentum_mag = float(np.hypot(vx_final, vy_final))
        clicks = float(sum(1 for step in trajectory if step['type'] in ['click_down', 'click_up']) / 256.0)
        avg_dt = float((sum(step['dt_ms'] for step in trajectory) / len(trajectory)) / 100.0)
        scaled_dwell = max(0.01, min(1.0, float(dwell_time_sec)))

        if getattr(self.simulator, 'context_dim', 8) == 8:
            return (
                float(vx_final),
                float(vy_final),
                float(scaled_dwell),
                float(jump_offset[0] / self.screen_w),
                float(jump_offset[1] / self.screen_h),
                float(clicks),
                float(avg_dt),
                float(momentum_mag)
            )
        else:
            return (float(vx_final), float(vy_final), float(clicks), float(avg_dt))

    def _update_context(self, trajectory, dwell_time_sec=0.1):
        """Updates internal momentum context vector from the most recent movement."""
        self.prev_context = self.compute_terminal_momentum(trajectory, dwell_time_sec=dwell_time_sec)

    def set_context(self, vx=0.0, vy=0.0, dwell_time=0.1, clicks=0.0, avg_dt=0.1):
        """
        Manually sets or overrides the velocity/momentum context vector.
        """
        mag = float(np.hypot(vx, vy))
        if getattr(self.simulator, 'context_dim', 8) == 8:
            self.prev_context = (float(vx), float(vy), float(dwell_time), 0.0, 0.0, float(clicks), float(avg_dt), mag)
        else:
            self.prev_context = (float(vx), float(vy), float(clicks), float(avg_dt))

    def get_context(self):
        """Returns current context vector."""
        return self.prev_context

    def move_to(self, target_x, target_y, click=False, button='left', delay_after=0.1, prev_context=None, target_radius=5.0, dwell_time=0.1):
        """
        Smoothly moves mouse from current position to (target_x, target_y) naturally using AI.
        prev_context: Optional context tuple to override velocity context.
        target_radius: Distance tolerance (px) to consider destination reached.
        dwell_time: Inter-waypoint dwell time (sec) for kinematic chaining.
        """
        start_x, start_y = self.get_current_position()
        ctx = prev_context if prev_context is not None else self.prev_context
        trajectory = self.generate_trajectory((start_x, start_y), (target_x, target_y), prev_context=ctx, intent=1.0, target_radius=target_radius)

        self._execute_trajectory(trajectory, perform_click=click, button=button)
        self._update_context(trajectory, dwell_time_sec=dwell_time)

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
        self._update_context(trajectory, dwell_time_sec=0.2)

        if delay_after > 0:
            time.sleep(delay_after)

        return trajectory

    def move_sequence(self, target_list, click_targets=False, delay_between=0.05, target_radius=5.0, continuous_chaining=True):
        """
        Moves mouse smoothly across a sequence of targets (x, y) maintaining continuous
        kinematic momentum chaining to avoid sharp geometric/polygonal turns.
        """
        results = []
        for i, target in enumerate(target_list):
            target_x, target_y = target
            is_last = (i == len(target_list) - 1)
            dwell = 0.20 if is_last else (0.04 if continuous_chaining else 0.15)
            traj = self.move_to(
                target_x,
                target_y,
                click=(click_targets if is_last else False),
                delay_after=(delay_between if not is_last else 0.0),
                target_radius=target_radius,
                dwell_time=dwell
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
