"""
Fixed & robust SUMO + TraCI RL script.
Edit the two variables below to match your system:
  - SUMO_CONFIG_PATH : absolute path to your RL.sumocfg file
  - SUMO_HOME        : if not already defined in the environment, set it here (optional)

Notes:
  * This script validates file paths before starting SUMO.
  * It handles missing detectors gracefully (returns 0 queue) so SUMO won't crash the Python loop.
  * Make sure detector IDs and traffic light id ("Node2") match those in your .net.xml / .sumocfg.
  * Adjust TOTAL_STEPS, MIN_GREEN_STEPS, EPSILON, ALPHA, GAMMA as needed.
"""

import os
import sys
import random
import time
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# USER CONFIG - EDIT THESE
# -----------------------
SUMO_CONFIG_PATH = r"C:\Users\ASUS\Desktop\tracii\tracii\RL.sumocfg"   # <<--- set absolute path to RL.sumocfg
# Optional: if SUMO_HOME not in environment, you can set it here:
# os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"  # Example for Windows
# -----------------------

# ----------  Validate SUMO_HOME and add tools to PYTHONPATH ----------
if 'SUMO_HOME' not in os.environ:
    sys.exit("ERROR: Please declare environment variable 'SUMO_HOME' (or set it at top of this script).")
sumo_tools_dir = os.path.join(os.environ['SUMO_HOME'], 'tools')
if not os.path.isdir(sumo_tools_dir):
    sys.exit(f"ERROR: SUMO tools directory not found at: {sumo_tools_dir}")
sys.path.append(sumo_tools_dir)

# ----------  Validate SUMO config exists ----------
if not os.path.isfile(SUMO_CONFIG_PATH):
    sys.exit(f"ERROR: SUMO config file not found: {SUMO_CONFIG_PATH}\nSet SUMO_CONFIG_PATH to the absolute path to RL.sumocfg")

# Use full path to sumo-gui (or sumo) binary
if os.name == 'nt':
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui.exe')
else:
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')

if not os.path.isfile(sumo_binary):
    # try non-gui sumo as fallback
    alt = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    if os.path.isfile(alt):
        sumo_binary = alt
    else:
        sys.exit(f"ERROR: SUMO binary not found in {os.path.join(os.environ['SUMO_HOME'], 'bin')}")

# ----------  Import traci after SUMO tools added ----------
try:
    import traci
except Exception as e:
    sys.exit(f"ERROR importing traci: {e}")

# ---------- Sumo start configuration ----------
Sumo_config = [
    sumo_binary,
    "-c", SUMO_CONFIG_PATH,
    "--step-length", "0.10",
    # "--delay", "1000",  # delay is for GUI startup latency; usually not required here
    "--lateral-resolution", "0"
]

# -------------------------
# RL + simulation variables
# -------------------------
TOTAL_STEPS = 10000    # reduce for debug; increase later
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]       # 0 = keep, 1 = switch

# Q-table dictionary
Q_table = {}

# stability params
MIN_GREEN_STEPS = 10   # keep small for debugging; increase if needed
last_switch_step = -MIN_GREEN_STEPS

# helper: IDs used in get_state() - change if your detectors use different names
DETECTORS = {
    "Node1_2_EB_0": "Node1_2_EB_0",
    "Node1_2_EB_1": "Node1_2_EB_1",
    "Node1_2_EB_2": "Node1_2_EB_2",
    "Node2_7_SB_0": "Node2_7_SB_0",
    "Node2_7_SB_1": "Node2_7_SB_1",
    "Node2_7_SB_2": "Node2_7_SB_2",
}
TRAFFIC_LIGHT_ID = "Node2"   # change if your TLS id differs

# -------------------------
# Utility & RL functions
# -------------------------
def get_max_Q_value_of_state(s):
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return float(np.max(Q_table[s]))

def get_reward(state):
    # negative total queue (minimize queues)
    total_queue = sum(state[:-1])  # last element is current_phase
    return -float(total_queue)

def safe_lanearea_getLastStepVehicleNumber(detector_id):
    """
    Safe wrapper for traci.lanearea.getLastStepVehicleNumber.
    If detector id is invalid or an exception occurs, returns 0.
    """
    try:
        return int(traci.lanearea.getLastStepVehicleNumber(detector_id))
    except Exception:
        return 0

def safe_trafficlight_getPhase(tls_id):
    try:
        return int(traci.trafficlight.getPhase(tls_id))
    except Exception:
        return 0

def get_state():
    """
    Returns a tuple: (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)
    Uses safe wrappers so missing detectors won't crash the script.
    """
    q_EB_0 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node1_2_EB_0"])
    q_EB_1 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node1_2_EB_1"])
    q_EB_2 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node1_2_EB_2"])
    q_SB_0 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node2_7_SB_0"])
    q_SB_1 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node2_7_SB_1"])
    q_SB_2 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node2_7_SB_2"])
    current_phase = safe_trafficlight_getPhase(TRAFFIC_LIGHT_ID)
    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)

def apply_action(action, tls_id=TRAFFIC_LIGHT_ID, current_simulation_step=0):
    """
    Performs action:
      0 -> keep current phase
      1 -> switch to next phase if MIN_GREEN_STEPS satisfied
    """
    global last_switch_step
    if action == 0:
        return
    elif action == 1:
        try:
            if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
                program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                num_phases = len(program.phases)
                next_phase = (safe_trafficlight_getPhase(tls_id) + 1) % num_phases
                traci.trafficlight.setPhase(tls_id, next_phase)
                last_switch_step = current_simulation_step
        except Exception:
            # If traffic light id or API calls fail, just ignore the switch for now
            return

def update_Q_table(old_state, action, reward, new_state):
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    old_q = float(Q_table[old_state][action])
    best_future_q = get_max_Q_value_of_state(new_state)
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def get_action_from_policy(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        return int(np.argmax(Q_table[state]))

# -------------------------
# Run SUMO + RL loop
# -------------------------
try:
    print("Starting SUMO via TraCI ...")
    traci.start(Sumo_config)
    # optional: set a GUI schema if you know the view id (may raise if not available)
    try:
        traci.gui.setSchema("View #0", "real world")
    except Exception:
        pass

    step_history = []
    reward_history = []
    queue_history = []
    cumulative_reward = 0.0

    print("\n=== Starting Fully Online Continuous Learning ===")
    for step in range(TOTAL_STEPS):
        current_simulation_step = step

        # Observe state
        state = get_state()
        action = get_action_from_policy(state)

        # Apply before simulationStep() if you want action to take effect in this step.
        apply_action(action, current_simulation_step=current_simulation_step)

        # Advance SUMO by 1 step
        traci.simulationStep()

        # Observe new state & reward
        new_state = get_state()
        reward = get_reward(new_state)
        cumulative_reward += reward

        # Update Q
        update_Q_table(state, action, reward, new_state)

        # Logging - adjust frequency if too chatty
        if step % 1 == 0:
            print(f"Step {step}, State={state}, Action={action}, NewState={new_state}, Reward={reward:.2f}, Cumulative={cumulative_reward:.2f}")
            step_history.append(step)
            reward_history.append(cumulative_reward)
            queue_history.append(sum(new_state[:-1]))

    print("\nSimulation finished, closing TraCI...")
except Exception as e:
    print(f"Exception during simulation: {e}")
finally:
    # Ensure TraCI closes cleanly
    try:
        traci.close()
    except Exception:
        pass

# -------------------------
# After-run reporting
# -------------------------
print("\nOnline Training completed. Final Q-table size:", len(Q_table))
for st, actions in Q_table.items():
    print("State:", st, "-> Q-values:", actions)

# -------------------------
# Visualization (if running in environment with display)
# -------------------------
if len(step_history) > 0:
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(step_history, reward_history, marker='o', linestyle='-')
        plt.xlabel("Simulation Step")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward over Steps")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(step_history, queue_history, marker='o', linestyle='-')
        plt.xlabel("Simulation Step")
        plt.ylabel("Total Queue Length")
        plt.title("Queue Length over Steps")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("Plotting failed (likely no display available):", e)
