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
    def __init__(self, noise_dim=4, hidden_dim=256):
        super(ConditioningEncoder, self).__init__()
        in_dim = 9 + noise_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, start_pos, target_pos, prev_context, intent, noise):
        x = torch.cat([start_pos, target_pos, prev_context, intent, noise], dim=-1)
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

    def forward(self, start_pos, target_pos, prev_context, intent, z_noise=None):
        batch_size = start_pos.size(0)
        if z_noise is None:
            z_noise = torch.randn(batch_size, self.noise_dim, device=start_pos.device)

        h0 = self.encoder(start_pos, target_pos, prev_context, intent, z_noise)
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

# =====================================================================
# Trajectory Simulation Engine
# =====================================================================

class HumanMouseSimulator:
    def __init__(self, model_path="human_mouse_model.pt", display_w=1920, display_h=1080):
        self.display_w = float(display_w)
        self.display_h = float(display_h)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HumanMouseGenerator(seq_len=256).to(self.device)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained AI mouse model from: {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained weights.")

        self.model.eval()

    def generate_trajectory(self, start_pos, target_pos, prev_context=None, intent=1.0):
        sx_norm = start_pos[0] / self.display_w
        sy_norm = start_pos[1] / self.display_h
        tx_norm = target_pos[0] / self.display_w
        ty_norm = target_pos[1] / self.display_h

        start_t = torch.tensor([[sx_norm, sy_norm]], dtype=torch.float32, device=self.device)
        target_t = torch.tensor([[tx_norm, ty_norm]], dtype=torch.float32, device=self.device)
        intent_t = torch.tensor([[intent]], dtype=torch.float32, device=self.device)

        if prev_context is None:
            ctx_t = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device)
        else:
            ctx_t = torch.tensor([[prev_context[0], prev_context[1], prev_context[2], prev_context[3]]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred_seq, pred_traj = self.model(start_t, target_t, ctx_t, intent_t)

        pred_seq = pred_seq[0].cpu().numpy()
        pred_traj = pred_traj[0].cpu().numpy()

        trajectory = []
        trajectory.append({
            "x": int(round(start_pos[0])),
            "y": int(round(start_pos[1])),
            "dt_ms": 0.0,
            "type": "move"
        })

        target_x, target_y = target_pos[0], target_pos[1]
        start_x, start_y = start_pos[0], start_pos[1]

        for step in range(len(pred_traj)):
            raw_x = pred_traj[step, 0] * self.display_w
            raw_y = pred_traj[step, 1] * self.display_h

            if step < 5:
                alpha = (step + 1) / 5.0
                x_px = (1 - alpha) * start_x + alpha * raw_x
                y_px = (1 - alpha) * start_y + alpha * raw_y
            else:
                x_px = raw_x
                y_px = raw_y

            dt_scaled = pred_seq[step, 2] * 100.0
            action = pred_seq[step, 3]

            dt_ms = max(4.0, min(float(dt_scaled), 20.0))

            action_type = "move"
            if action >= 0.8 and action < 1.5:
                action_type = "click_down"
            elif action >= 1.5 and action < 2.5:
                action_type = "click_up"

            trajectory.append({
                "x": int(round(x_px)),
                "y": int(round(y_px)),
                "dt_ms": round(dt_ms, 2),
                "type": action_type
            })

            dist_to_target = np.hypot(target_x - x_px, target_y - y_px)
            if dist_to_target < 15.0 and step > 10:
                break

        trajectory.append({
            "x": int(round(target_x)),
            "y": int(round(target_y)),
            "dt_ms": 8.0,
            "type": "click_down"
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
    def __init__(self, model_path=None, failsafe=True):
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

    def generate_trajectory(self, start_pos, target_pos, prev_context=None, intent=1.0):
        """Generates AI trajectory sequence without executing physical mouse movement."""
        return self.simulator.generate_trajectory(
            start_pos=start_pos,
            target_pos=target_pos,
            prev_context=prev_context if prev_context is not None else self.prev_context,
            intent=intent
        )

    def _execute_trajectory(self, trajectory, perform_click=False, button='left'):
        """Executes generated trajectory points with high-precision timing."""
        if not trajectory:
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

    def move_to(self, target_x, target_y, click=False, button='left', delay_after=0.1, prev_context=None):
        """
        Smoothly moves mouse from current position to (target_x, target_y) naturally using AI.
        prev_context: Optional 4D tuple (vx, vy, clicks, avg_dt) to override velocity context.
        """
        start_x, start_y = self.get_current_position()
        ctx = prev_context if prev_context is not None else self.prev_context
        trajectory = self.generate_trajectory((start_x, start_y), (target_x, target_y), prev_context=ctx, intent=1.0)

        self._execute_trajectory(trajectory, perform_click=click, button=button)
        self._update_context(trajectory)

        if delay_after > 0:
            time.sleep(delay_after)

        return trajectory

    def click_at(self, target_x, target_y, button='left', delay_after=0.1, prev_context=None):
        """Moves mouse naturally to (target_x, target_y) and clicks."""
        return self.move_to(target_x, target_y, click=True, button=button, delay_after=delay_after, prev_context=prev_context)

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

    def move_sequence(self, target_list, click_targets=False, delay_between=0.4):
        """Moves mouse smoothly across a sequence of targets (x, y) maintaining momentum context."""
        results = []
        for i, target in enumerate(target_list):
            target_x, target_y = target
            traj = self.move_to(
                target_x,
                target_y,
                click=click_targets,
                delay_after=delay_between if i < len(target_list) - 1 else 0.0
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
