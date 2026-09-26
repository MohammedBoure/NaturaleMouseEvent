import os
import sys
import numpy as np
import torch
from human_mouse import HumanMouseSimulator

if __name__ == '__main__':
    print("\n--- AI Natural Mouse Trajectory Simulation Test ---")
    model_file = "human_mouse_model.pt"
    if not os.path.exists(model_file) and os.path.exists("best_model.pth"):
        model_file = "best_model.pth"

    sim = HumanMouseSimulator(model_path=model_file)
    start_pos = (100, 100)
    target_pos = (800, 500)
    print(f"Simulating movement: {start_pos} -> {target_pos}")
    traj = sim.generate_trajectory(start_pos, target_pos)

    print(f"\nGenerated AI Trajectory: {len(traj)} steps")
    print("First 3 steps:")
    for s in traj[:3]:
        print(f"  {s}")
    print("Last 3 steps:")
    for s in traj[-3:]:
        print(f"  {s}")

    total_time_ms = sum(s['dt_ms'] for s in traj)
    final_pos = (traj[-1]['x'], traj[-1]['y'])
    dist_error = np.hypot(final_pos[0] - target_pos[0], final_pos[1] - target_pos[1])
    print(f"\nSummary:")
    print(f"  Total Duration: {total_time_ms:.1f} ms")
    print(f"  Final Position: {final_pos} (Target: {target_pos})")
    print(f"  Final Distance to Target: {dist_error:.2f} px")
    print("---------------------------------------------------\n")
