import torch
import numpy as np
import os
import json
from train_model import HumanMouseGenerator

class HumanMouseSimulator:
    def __init__(self, model_path=r"c:\Users\moham\Desktop\ai\naturale_mouse_event\human_mouse_model.pt", display_w=1920, display_h=1080):
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

    def generate_trajectory(self, start_pos, target_pos, prev_context=None, max_steps=256):
        """
        Generate natural human mouse trajectory.
        start_pos: (x, y) absolute pixels
        target_pos: (x, y) absolute pixels
        prev_context: (vx, vy) normalized velocity context
        Returns: list of dicts [{'x': int, 'y': int, 'dt_ms': float, 'type': str}]
        """
        sx_norm = start_pos[0] / self.display_w
        sy_norm = start_pos[1] / self.display_h
        tx_norm = target_pos[0] / self.display_w
        ty_norm = target_pos[1] / self.display_h

        start_t = torch.tensor([[sx_norm, sy_norm]], dtype=torch.float32, device=self.device)
        target_t = torch.tensor([[tx_norm, ty_norm]], dtype=torch.float32, device=self.device)

        if prev_context is None:
            ctx_t = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device)
        else:
            ctx_t = torch.tensor([[prev_context[0], prev_context[1], prev_context[2], prev_context[3]]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred_seq, pred_traj = self.model(start_t, target_t, ctx_t)

        pred_seq = pred_seq[0].cpu().numpy() # (256, 4) -> [dx, dy, dt, action]
        pred_traj = pred_traj[0].cpu().numpy() # (256, 2) -> [x_norm, y_norm]

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
            # Direct cumulative coordinates from neural network trajectory output
            raw_x = pred_traj[step, 0] * self.display_w
            raw_y = pred_traj[step, 1] * self.display_h

            # Smoothly blend first 5 steps to anchor to start_pos
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

            # Stop when close enough to target point
            dist_to_target = np.hypot(target_x - x_px, target_y - y_px)
            if dist_to_target < 15.0 and step > 10:
                break

        # Snap final step precisely to target position with click
        trajectory.append({
            "x": int(round(target_x)),
            "y": int(round(target_y)),
            "dt_ms": 8.0,
            "type": "click_down"
        })

        return trajectory

if __name__ == '__main__':
    sim = HumanMouseSimulator()
    traj = sim.generate_trajectory((100, 100), (800, 500))
    print(f"Generated AI Trajectory: {len(traj)} steps")
    print("First 3 steps:", traj[:3])
    print("Last 3 steps:", traj[-3:])
