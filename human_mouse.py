import time
import os
import sys
import numpy as np
from simulate_mouse import HumanMouseSimulator

# Enable UTF-8 console printing on Windows
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

class HumanMouse:
    """
    High-level Programmatic API for natural AI human mouse movement.
    Uses trained PyTorch neural network weights to generate realistic human mouse trajectories.
    """
    def __init__(self, model_path=None, failsafe=True):
        if model_path is None:
            default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_mouse_model.pt")
            model_path = default_path if os.path.exists(default_path) else "human_mouse_model.pt"

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

    def generate_trajectory(self, start_pos, target_pos, prev_context=None):
        """
        Generates AI trajectory sequence without executing physical mouse movement.
        Useful for testing, visualization, or headless environments.
        """
        return self.simulator.generate_trajectory(
            start_pos=start_pos,
            target_pos=target_pos,
            prev_context=prev_context if prev_context is not None else self.prev_context
        )

    def _execute_trajectory(self, trajectory, perform_click=False, button='left'):
        """
        Executes generated trajectory points with high-precision timing and low CPU usage.
        """
        if not trajectory:
            return

        if PYAUTOGUI_AVAILABLE:
            start_time = time.perf_counter()
            cumulative_target_sec = 0.0

            for step in trajectory:
                dt_sec = step['dt_ms'] / 1000.0
                cumulative_target_sec += dt_sec

                # Hybrid Sleep & Spin-wait for precise timing
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
                except pyautogui.FailSafeException:
                    pass

            if perform_click:
                try:
                    pyautogui.click(button=button)
                except pyautogui.FailSafeException:
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

    def move_to(self, target_x, target_y, click=False, button='left', delay_after=0.1):
        """
        Smoothly moves mouse from current position to (target_x, target_y) naturally using AI.
        """
        start_x, start_y = self.get_current_position()
        trajectory = self.generate_trajectory((start_x, start_y), (target_x, target_y))

        self._execute_trajectory(trajectory, perform_click=click, button=button)
        self._update_context(trajectory)

        if delay_after > 0:
            time.sleep(delay_after)

        return trajectory

    def click_at(self, target_x, target_y, button='left', delay_after=0.1):
        """
        Moves mouse naturally to (target_x, target_y) and clicks.
        """
        return self.move_to(target_x, target_y, click=True, button=button, delay_after=delay_after)

    def move_sequence(self, target_list, click_targets=False, delay_between=0.4):
        """
        Moves mouse smoothly across a sequence of targets (x, y) maintaining momentum context.
        """
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
    print("Testing HumanMouse module dry-run...")
    mouse = HumanMouse()
    pos = mouse.get_current_position()
    print(f"Current Position: {pos}")
    traj = mouse.generate_trajectory(pos, (500, 500))
    print(f"Generated trajectory length: {len(traj)} steps")
