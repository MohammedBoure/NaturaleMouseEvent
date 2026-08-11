"""
Example Script: Demonstrates programmatic usage of the HumanMouse AI model.
Can be imported and integrated into Selenium, Playwright, PyAutoGUI, or bot automation scripts.
"""

import sys
from human_mouse import HumanMouse

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def example_usage():
    print("--- Initialize Human Mouse AI Engine ---")
    mouse = HumanMouse(failsafe=True)
    print(f"Screen Resolution: {mouse.screen_w}x{mouse.screen_h}")

    # Example 1: Move mouse naturally to position (800, 400)
    print("\n1. Moving mouse naturally to (800, 400)...")
    mouse.move_to(800, 400)

    # Example 2: Move mouse naturally and click at (1200, 600)
    print("\n2. Moving mouse naturally and clicking at (1200, 600)...")
    mouse.click_at(1200, 600, button='left')

    # Example 3: Execute a continuous sequence of natural movements
    print("\n3. Executing a sequence of targets...")
    sequence = [(400, 300), (900, 500), (600, 800)]
    mouse.move_sequence(sequence, click_targets=False, delay_between=0.3)

    # Example 4: Generate trajectory data without moving physical mouse (useful for simulation/testing)
    print("\n4. Generating raw trajectory data for offline analysis...")
    trajectory = mouse.generate_trajectory(start_pos=(100, 100), target_pos=(500, 500))
    print(f"Generated {len(trajectory)} trajectory points.")
    print("Sample step:", trajectory[10])

    print("\n✅ Programmatic API demo completed successfully!")

if __name__ == '__main__':
    example_usage()
