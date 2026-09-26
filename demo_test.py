"""
Interactive Demonstration and Testing Suite for NaturaleMouseEvent.
Author: Senior AI Kinematics & Robotics Engineer
Project: NaturaleMouseEvent
"""

import os
import sys
import time
import math
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np

# Enable UTF-8 encoding on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from human_mouse import HumanMouse, PYAUTOGUI_AVAILABLE

try:
    import matplotlib
    matplotlib.use('Agg')  # Headless rendering compatible with all environments
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def print_banner():
    print("""
========================================================================
   🖱️  NATURALE MOUSE EVENT: AI INTERACTIVE DEMO & TESTING SUITE
   Biologically Authentic Human Cursor Kinematics & Neural Simulation
========================================================================
""")


def run_countdown(seconds: int = 3, message: str = "Starting in"):
    """Displays a non-intrusive countdown to allow the user to release their mouse."""
    print(f"\n[Notice] {message}:", end=" ", flush=True)
    for i in range(seconds, 0, -1):
        print(f"{i}...", end=" ", flush=True)
        time.sleep(1.0)
    print("GO!\n")


def run_dry_run_benchmark(mouse: HumanMouse, num_trials: int = 5):
    """
    Executes simulated trajectories without physical cursor movement
    and computes statistical accuracy, duration, and reach errors.
    """
    print(f"\n--- [1] DRY-RUN BENCHMARK ({num_trials} Trials) ---")
    print("Testing neural trajectory generator across varied screen distances...")

    sw, sh = mouse.screen_w, mouse.screen_h
    reach_errors = []
    durations = []
    step_counts = []

    print(f"{'Trial':<7} | {'Start Pos':<14} | {'Target Pos':<14} | {'Distance':<10} | {'Steps':<7} | {'Duration':<10} | {'Reach Error':<12}")
    print("-" * 85)

    for i in range(1, num_trials + 1):
        # Generate varied start & target pairs
        s_x = float(np.random.randint(int(sw * 0.1), int(sw * 0.9)))
        s_y = float(np.random.randint(int(sh * 0.1), int(sh * 0.9)))
        t_x = float(np.random.randint(int(sw * 0.1), int(sw * 0.9)))
        t_y = float(np.random.randint(int(sh * 0.1), int(sh * 0.9)))
        nominal_dist = np.hypot(t_x - s_x, t_y - s_y)

        traj = mouse.generate_trajectory(
            start_pos=(s_x, s_y),
            target_pos=(t_x, t_y),
            intent=1.0,
            target_radius=5.0
        )

        final_pt = traj[-1]
        err = np.hypot(final_pt['x'] - t_x, final_pt['y'] - t_y)
        total_time = sum(pt['dt_ms'] for pt in traj)

        reach_errors.append(err)
        durations.append(total_time)
        step_counts.append(len(traj))

        print(f"#{i:<6} | ({int(s_x):>4}, {int(s_y):>4}) | ({int(t_x):>4}, {int(t_y):>4}) | {nominal_dist:>6.1f} px | {len(traj):<7} | {total_time:>6.1f} ms | {err:>5.2f} px")

    mean_err = np.mean(reach_errors)
    mean_dur = np.mean(durations)
    mean_steps = np.mean(step_counts)

    print("-" * 85)
    print(f"📊 SUMMARY: Avg Reach Error: {mean_err:.2f} px | Avg Duration: {mean_dur:.1f} ms | Avg Steps: {mean_steps:.1f}")
    print("✅ Benchmark completed successfully!\n")


def run_live_cursor_demo(mouse: HumanMouse):
    """
    Executes live cursor movement across 4 screen quadrants,
    demonstrating momentum preservation, natural curvature, and easing.
    """
    print("\n--- [2] LIVE ON-SCREEN CURSOR DEMO ---")
    print("This mode will physically move your mouse cursor across 4 screen waypoints.")
    print("Safety Reminder: Move your mouse to (0, 0) top-left at any time to abort.")

    run_countdown(seconds=3, message="Cursor control will begin in")

    sw, sh = mouse.screen_w, mouse.screen_h
    waypoints = [
        ("Screen Center", (int(sw * 0.5), int(sh * 0.5))),
        ("Top-Right Box", (int(sw * 0.8), int(sh * 0.25))),
        ("Bottom-Right Box", (int(sw * 0.75), int(sh * 0.75))),
        ("Bottom-Left Box", (int(sw * 0.25), int(sh * 0.75))),
        ("Center Rest", (int(sw * 0.5), int(sh * 0.5))),
    ]

    for label, target in waypoints:
        curr = mouse.get_current_position()
        dist = np.hypot(target[0] - curr[0], target[1] - curr[1])
        print(f" ➔ Moving to {label} at {target} (Distance: {dist:.1f} px)...")
        start_t = time.perf_counter()
        mouse.move_to(target[0], target[1])
        elapsed = (time.perf_counter() - start_t) * 1000.0
        final_pos = mouse.get_current_position()
        err = np.hypot(final_pos[0] - target[0], final_pos[1] - target[1])
        print(f"   ✓ Arrived at {final_pos} | Elapsed: {elapsed:.1f} ms | Reach error: {err:.1f} px")
        time.sleep(0.3)

    print("\n✅ Live movement sequence completed gracefully!\n")


def run_precision_click_demo(mouse: HumanMouse):
    """
    Demonstrates moving towards a target and executing a natural click
    with biological deceleration and submovement fine-tuning.
    """
    print("\n--- [3] PRECISION TARGET CLICK DEMO ---")
    print("The cursor will move toward a designated target and perform an authentic human click.")

    run_countdown(seconds=3, message="Starting in")

    sw, sh = mouse.screen_w, mouse.screen_h
    target_x = int(sw * 0.6)
    target_y = int(sh * 0.4)

    print(f" ➔ Navigating and clicking target at ({target_x}, {target_y})...")
    mouse.click(target_x, target_y)
    final_pos = mouse.get_current_position()
    err = np.hypot(final_pos[0] - target_x, final_pos[1] - target_y)
    print(f"   ✓ Click executed at {final_pos} | Reach error: {err:.1f} px\n")


def run_idle_wandering_demo(mouse: HumanMouse):
    """
    Demonstrates human-like idle wandering and micro-tremor around the current position.
    """
    print("\n--- [4] IDLE WANDERING & MICRO-TREMOR DEMO ---")
    print("Demonstrating biological hesitation and natural cursor wandering behavior...")

    run_countdown(seconds=2, message="Wandering starts in")

    curr = mouse.get_current_position()
    print(f" ➔ Wandering around current location {curr} for 3 sub-movements...")
    for step in range(3):
        mouse.wander(radius=120)
        time.sleep(0.2)

    print("✅ Idle wandering completed!\n")


def run_trajectory_plot_demo(mouse: HumanMouse, output_file: str = "trajectory_demo.png"):
    """
    Generates a comprehensive kinematic trajectory analysis plot:
    1. 2D Screen Space Path (with velocity color gradient)
    2. Bell-shaped Velocity Profile (v(t) vs time ms)
    3. Acceleration & Deceleration profile showing natural easing
    """
    print(f"\n--- [5] KINEMATIC TRAJECTORY PLOTTER ---")
    if not MATPLOTLIB_AVAILABLE:
        print("[Warning] 'matplotlib' is not installed. Skipping plot generation.")
        return

    sw, sh = mouse.screen_w, mouse.screen_h
    start_pos = (int(sw * 0.15), int(sh * 0.2))
    target_pos = (int(sw * 0.75), int(sh * 0.7))

    print(f"Generating neural trajectory: {start_pos} -> {target_pos}...")
    traj = mouse.generate_trajectory(start_pos, target_pos, intent=1.0)

    # Extract coordinates and times
    xs = np.array([pt['x'] for pt in traj], dtype=np.float32)
    ys = np.array([pt['y'] for pt in traj], dtype=np.float32)
    dts = np.array([pt['dt_ms'] for pt in traj], dtype=np.float32)
    timestamps = np.cumsum(dts)

    # Compute continuous kinematics
    dx = np.diff(xs)
    dy = np.diff(ys)
    step_dists = np.hypot(dx, dy)
    dt_mid = dts[1:]
    velocities_px_per_s = np.zeros_like(step_dists)
    valid_dt = dt_mid > 1e-3
    velocities_px_per_s[valid_dt] = (step_dists[valid_dt] / (dt_mid[valid_dt] / 1000.0))

    accelerations = np.zeros_like(velocities_px_per_s)
    if len(velocities_px_per_s) > 1:
        dv = np.diff(velocities_px_per_s)
        dt_acc = dt_mid[1:]
        valid_acc = dt_acc > 1e-3
        accelerations[1:][valid_acc] = dv[valid_acc] / (dt_acc[valid_acc] / 1000.0)

    # Plot figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#1e1e1e')

    # Subplot 1: 2D Spatial Trajectory
    ax1 = axes[0]
    ax1.set_facecolor('#121212')
    scatter = ax1.scatter(xs[:-1], ys[:-1], c=velocities_px_per_s, cmap='viridis', s=12, alpha=0.85)
    ax1.plot(xs, ys, color='#58a6ff', linewidth=1.2, alpha=0.6, label='Trajectory Path')
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start Point')
    ax1.plot(target_pos[0], target_pos[1], 'r*', markersize=14, label='Target Point')
    ax1.plot(xs[-1], ys[-1], 'mo', markersize=8, fillstyle='none', markeredgewidth=2, label='Final Landing')

    # Draw straight reference line
    ax1.plot([start_pos[0], target_pos[0]], [start_pos[1], target_pos[1]], 'w--', alpha=0.25, label='Direct Vector')

    ax1.set_title("2D Spatial Trajectory (Biological Arc)", color='white', fontsize=12, fontweight='bold')
    ax1.set_xlabel("Screen X (px)", color='#c9d1d9')
    ax1.set_ylabel("Screen Y (px)", color='#c9d1d9')
    ax1.tick_params(colors='#8b949e')
    ax1.invert_yaxis()  # Match screen coordinates (0, 0 top-left)
    ax1.grid(True, color='#30363d', linestyle=':', alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Velocity (px/s)', color='#c9d1d9')
    cbar.ax.yaxis.set_tick_params(color='#8b949e')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8b949e')
    ax1.legend(loc='lower right', facecolor='#21262d', edgecolor='#30363d', labelcolor='white')

    # Subplot 2: Velocity Profile (Fitts's Law Bell Curve)
    ax2 = axes[1]
    ax2.set_facecolor('#121212')
    ax2.plot(timestamps[1:], velocities_px_per_s, color='#3fb950', linewidth=2.0)
    ax2.fill_between(timestamps[1:], velocities_px_per_s, color='#238636', alpha=0.3)
    ax2.set_title("Bell-Shaped Velocity Profile v(t)", color='white', fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (ms)", color='#c9d1d9')
    ax2.set_ylabel("Velocity (px/s)", color='#c9d1d9')
    ax2.tick_params(colors='#8b949e')
    ax2.grid(True, color='#30363d', linestyle=':', alpha=0.6)

    # Subplot 3: Acceleration & Deceleration Dynamics
    ax3 = axes[2]
    ax3.set_facecolor('#121212')
    ax3.plot(timestamps[1:], accelerations, color='#f78166', linewidth=1.5)
    ax3.axhline(0, color='#8b949e', linestyle='--', alpha=0.5)
    ax3.set_title("Acceleration & Deceleration (Easing Dynamics)", color='white', fontsize=12, fontweight='bold')
    ax3.set_xlabel("Time (ms)", color='#c9d1d9')
    ax3.set_ylabel("Acceleration (px/s²)", color='#c9d1d9')
    ax3.tick_params(colors='#8b949e')
    ax3.grid(True, color='#30363d', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    print(f"📊 Visualization successfully exported to: {os.path.abspath(output_file)}")
    print(f"   • Total movement duration: {timestamps[-1]:.1f} ms")
    print(f"   • Peak velocity: {np.max(velocities_px_per_s):.1f} px/s")
    print(f"   • Endpoint reach error: {np.hypot(xs[-1] - target_pos[0], ys[-1] - target_pos[1]):.2f} px\n")


def interactive_menu():
    """Renders interactive CLI selection menu."""
    print_banner()

    mouse = HumanMouse()
    print(f"Display Dimensions: {mouse.screen_w}x{mouse.screen_h} px")
    print(f"Current Cursor Position: {mouse.get_current_position()}")

    while True:
        print("""
Select a demonstration mode:
 [1] Dry-Run Accuracy Benchmark (Simulation Only - Cursor Does NOT Move)
 [2] Live On-Screen Cursor Test (Moves Cursor Across 4 Waypoints)
 [3] Precision Target Click Test (Moves & Clicks a Single Target)
 [4] Idle Wandering & Physiological Tremor Test
 [5] Kinematic Trajectory Plotter (Exports trajectory_demo.png analysis graph)
 [6] Run All Tests Sequentially
 [0] Exit
""")
        choice = input("Enter option [0-6]: ").strip()

        if choice == '1':
            run_dry_run_benchmark(mouse)
        elif choice == '2':
            run_live_cursor_demo(mouse)
        elif choice == '3':
            run_precision_click_demo(mouse)
        elif choice == '4':
            run_idle_wandering_demo(mouse)
        elif choice == '5':
            run_trajectory_plot_demo(mouse)
        elif choice == '6':
            run_dry_run_benchmark(mouse)
            run_trajectory_plot_demo(mouse)
            run_live_cursor_demo(mouse)
            run_precision_click_demo(mouse)
            run_idle_wandering_demo(mouse)
        elif choice == '0':
            print("Exiting demo suite. Happy scripting!")
            break
        else:
            print("Invalid selection. Please choose an option from 0 to 6.")


def main():
    parser = argparse.ArgumentParser(description="NaturaleMouseEvent Interactive Demo Suite")
    parser.add_argument("--mode", type=str, choices=["dry-run", "live", "click", "wander", "plot", "all"],
                        help="Direct execution mode without interactive prompt")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials for benchmark")
    parser.add_argument("--plot-out", type=str, default="trajectory_demo.png", help="Output path for plot")

    args = parser.parse_args()

    if args.mode:
        print_banner()
        mouse = HumanMouse()
        if args.mode == "dry-run":
            run_dry_run_benchmark(mouse, num_trials=args.trials)
        elif args.mode == "live":
            run_live_cursor_demo(mouse)
        elif args.mode == "click":
            run_precision_click_demo(mouse)
        elif args.mode == "wander":
            run_idle_wandering_demo(mouse)
        elif args.mode == "plot":
            run_trajectory_plot_demo(mouse, output_file=args.plot_out)
        elif args.mode == "all":
            run_dry_run_benchmark(mouse, num_trials=args.trials)
            run_trajectory_plot_demo(mouse, output_file=args.plot_out)
            run_live_cursor_demo(mouse)
            run_precision_click_demo(mouse)
            run_idle_wandering_demo(mouse)
    else:
        interactive_menu()


if __name__ == '__main__':
    main()
