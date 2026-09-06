"""
STATIC / FIXED-TIME TRAFFIC SIGNAL CONTROLLER (BASELINE)

Key Characteristics (per project presentation Slide 11):
- Operates on fixed signal durations with no adaptive changes during simulation.
- Does NOT learn or adapt based on traffic flow.
- Follows the pre-programmed static signal cycle defined in SUMO network (Node2).
- Serves as the benchmark baseline against which RL models (Q-Learning & DQN) are evaluated.

ITS Metrics Collected:
- Total delay (veh-seconds)
- Average queue length (vehicles)
- Max queue length (vehicles)
- Vehicles completed (throughput)
- Average travel time per vehicle (s)
- 90th percentile travel time (s)
- Max travel time (s)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Dynamic Paths Resolution
# ---------------------------------------------------------
CONTROLLERS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONTROLLERS_DIR)
SUMO_CONFIG_PATH = os.path.join(PROJECT_ROOT, "sumo_config", "RL.sumocfg")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------
STEP_LENGTH = 1.0       # seconds per simulation step
TOTAL_STEPS = 3000      # total simulation steps

# Detectors and Traffic Light configuration
DETECTORS = [
    "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2",
    "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"
]
TRAFFIC_LIGHT_ID = "Node2"


def setup_sumo_environment():
    """Verify SUMO_HOME, locate SUMO tools and binary."""
    if 'SUMO_HOME' not in os.environ:
        default_sumo_paths = [
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Program Files\Eclipse\Sumo"
        ]
        found = False
        for p in default_sumo_paths:
            if os.path.isdir(p):
                os.environ['SUMO_HOME'] = p
                found = True
                break
        if not found:
            sys.exit("❌ ERROR: Please declare environment variable 'SUMO_HOME' before running.")

    sumo_tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if not os.path.isdir(sumo_tools):
        sys.exit(f"❌ ERROR: SUMO tools directory not found at: {sumo_tools}")
    if sumo_tools not in sys.path:
        sys.path.append(sumo_tools)

    if not os.path.isfile(SUMO_CONFIG_PATH):
        sys.exit(f"❌ ERROR: SUMO config file not found at: {SUMO_CONFIG_PATH}")

    if os.name == 'nt':
        sumo_gui_path = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui.exe')
    else:
        sumo_gui_path = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')

    if not os.path.isfile(sumo_gui_path):
        fallback = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
        if os.path.isfile(fallback):
            print("⚠️ sumo-gui not found — falling back to headless sumo binary.")
            return fallback
        sys.exit("❌ ERROR: Neither sumo-gui nor sumo binary found in SUMO_HOME/bin.")
    return sumo_gui_path


def compute_metrics(queue_history, travel_times, step_length=STEP_LENGTH):
    """Compute standard ITS metrics."""
    if queue_history:
        total_delay = sum(q * step_length for q in queue_history)
        avg_queue = float(sum(queue_history)) / len(queue_history)
        max_queue = max(queue_history)
    else:
        total_delay, avg_queue, max_queue = 0.0, 0.0, 0

    throughput = len(travel_times)

    if travel_times:
        arr = np.array(travel_times, dtype=np.float32)
        avg_tt = float(arr.mean())
        max_tt = float(arr.max())
        p90_tt = float(np.percentile(arr, 90))
    else:
        avg_tt, max_tt, p90_tt = 0.0, 0.0, 0.0

    return total_delay, avg_queue, max_queue, throughput, avg_tt, p90_tt, max_tt


def run():
    """Execute the Static Baseline Controller simulation."""
    sumo_binary = setup_sumo_environment()
    import traci

    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CONFIG_PATH,
        "--step-length", str(STEP_LENGTH),
        "--lateral-resolution", "0"
    ]

    print("\n========================================================")
    print("🚦 STARTING SUMO: STATIC / FIXED-TIME BASELINE CONTROLLER")
    print("========================================================")
    print(f"SUMO config: {SUMO_CONFIG_PATH}")
    print(f"Simulation steps: {TOTAL_STEPS} | Step length: {STEP_LENGTH}s")
    print("--------------------------------------------------------\n")

    step_history = []
    time_history = []
    queue_history = []
    depart_times = {}
    travel_times = []

    try:
        traci.start(sumo_cmd)
        print("✅ SUMO started successfully.")

        try:
            traci.gui.setSchema("View #0", "real world")
            traci.gui.setSpeed("View #0", 5)
        except Exception:
            pass

        for step in range(TOTAL_STEPS):
            current_time = step * STEP_LENGTH

            # Let SUMO run its pre-programmed static fixed-time signal plan
            traci.simulationStep()

            # Record detector vehicle counts (Queue length)
            current_queue = sum(traci.lanearea.getLastStepVehicleNumber(det) for det in DETECTORS)
            queue_history.append(current_queue)
            step_history.append(step)
            time_history.append(current_time)

            # Track vehicle travel times
            for vid in traci.simulation.getDepartedIDList():
                depart_times[vid] = current_time

            for vid in traci.simulation.getArrivedIDList():
                dt = current_time - depart_times.pop(vid, current_time)
                travel_times.append(dt)

            if step % 200 == 0:
                current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                print(f"Step {step:4d} | Time={current_time:6.1f}s | Phase={current_phase} | "
                      f"Queue={current_queue:2d} veh | Finished={len(travel_times)} veh")

            # Check if simulation completed early
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"\nAll expected vehicles cleared; stopping at step {step}.")
                break

    except Exception as e:
        print("❌ Simulation Error:", e)
    finally:
        try:
            traci.close()
            print("✅ SUMO connection closed cleanly.\n")
        except Exception:
            pass

    # ---------------------------------------------------------
    # Evaluation & ITS Metrics Reporting
    # ---------------------------------------------------------
    if len(time_history) > 0:
        total_delay, avg_queue, max_queue, throughput, avg_tt, p90_tt, max_tt = compute_metrics(
            queue_history, travel_times, STEP_LENGTH
        )

        print("\n" + "=" * 36)
        print("=== STATIC CONTROLLER METRICS ===")
        print("=" * 36)
        print(f"Total delay (veh-seconds):      {total_delay:.2f}")
        print(f"Average queue length (vehicles): {avg_queue:.2f}")
        print(f"Max queue length (vehicles):     {max_queue}")
        print(f"Vehicles completed (throughput): {throughput}")
        print(f"Average travel time / veh (s):   {avg_tt:.2f}")
        print(f"90th percentile travel time (s): {p90_tt:.2f}")
        print(f"Max travel time (s):             {max_tt:.2f}")
        print("=" * 36 + "\n")

        # Save results to outputs/
        npz_path = os.path.join(OUTPUTS_DIR, "results_static.npz")
        np.savez(
            npz_path,
            time=np.array(time_history, dtype=np.float32),
            queue=np.array(queue_history, dtype=np.float32),
            travel_times=np.array(travel_times, dtype=np.float32)
        )
        print(f"📁 Results saved to: {npz_path}")

        # Plot Queue Length over Time
        plt.figure(figsize=(10, 5))
        plt.plot(time_history, queue_history, color='tab:blue', label="Total Queue Length")
        plt.xlabel("Time (s)")
        plt.ylabel("Total Queue Length (vehicles)")
        plt.title("Static Controller: Queue Length Over Time")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plot_path = os.path.join(OUTPUTS_DIR, "static_queue_plot.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"📊 Plot saved to: {plot_path}")
        plt.show()


if __name__ == "__main__":
    run()
