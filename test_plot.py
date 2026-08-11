import torch
from simulate_mouse import HumanMouseSimulator
import matplotlib.pyplot as plt
import numpy as np

sim = HumanMouseSimulator()
traj = sim.generate_trajectory((100, 100), (800, 500))

xs = [p['x'] for p in traj]
ys = [p['y'] for p in traj]
dts = [p['dt_ms'] for p in traj]

print("Total steps:", len(traj))
print("Total time (ms):", sum(dts))

# Plot the trajectory
plt.figure(figsize=(10, 6))
plt.plot(xs, ys, marker='.', markersize=2)
plt.scatter([100], [100], color='green', label='Start')
plt.scatter([800], [500], color='red', label='Target')
plt.title(f"AI Mouse Trajectory - {len(traj)} steps, {sum(dts):.1f} ms")
plt.xlim(0, 1920)
plt.ylim(0, 1080)
plt.gca().invert_yaxis() # screen coords
plt.legend()
plt.savefig("trajectory_plot.png")
print("Saved trajectory_plot.png")

# Plot speed (px per ms)
speeds = []
for i in range(1, len(traj)):
    dx = xs[i] - xs[i-1]
    dy = ys[i] - ys[i-1]
    dist = np.hypot(dx, dy)
    dt = max(1.0, dts[i]) # avoid division by zero
    speeds.append(dist / dt)

plt.figure(figsize=(10, 4))
plt.plot(speeds)
plt.title("Mouse Speed (pixels / ms)")
plt.xlabel("Step")
plt.ylabel("Speed")
plt.savefig("speed_plot.png")
print("Saved speed_plot.png")
