"""
Robust SUMO + TraCI script — replacement for your provided code.

What I changed / ensured:
 - Validates SUMO_HOME and RL.sumocfg path up-front and fails with a clear message if missing.
 - Uses safe wrappers for detector and traffic light calls so missing IDs won't crash SUMO loop.
 - Keeps your RL hyperparameters/behaviour (EPSILON=0, deterministic policy returning action 0).
 - Offers reduced TOTAL_STEPS by default for quicker debug; set it back to 10000 if you want.
 - Uses absolute SUMO binary path (from SUMO_HOME) instead of just 'sumo-gui'.
 - Logs state and queue stats every 100 steps (matching your print frequency).
 - Closes traci cleanly in a finally block.
 
Edit these two variables near the top:
  SUMO_CONFIG_PATH = r"C:\full\path\to\RL.sumocfg"
  TOTAL_STEPS (if you want)

Make sure detector IDs and traffic light id match those in your SUMO files.
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
TOTAL_STEPS = 10000                                        # set back to 10000 if desired
# -----------------------

# -----------------------
# Validate SUMO_HOME and add tools to PYTHONPATH
# -----------------------
if 'SUMO_HOME' not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME' before running this script.")

sumo_tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if not os.path.isdir(sumo_tools):
    sys.exit(f"SUMO tools not found at expected location: {sumo_tools}")
sys.path.append(sumo_tools)

# -----------------------
# Validate sumo config and binary
# -----------------------
if not os.path.isfile(SUMO_CONFIG_PATH):
    sys.exit(f"SUMO config file not found: {SUMO_CONFIG_PATH}\nSet SUMO_CONFIG_PATH to the absolute path of RL.sumocfg")

if os.name == 'nt':
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui.exe')
else:
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')

if not os.path.isfile(sumo_binary):
    # fallback to non-gui binary
    alt = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    if os.path.isfile(alt):
        sumo_binary = alt
    else:
        sys.exit(f"SUMO binary not found in {os.path.join(os.environ['SUMO_HOME'], 'bin')}")

# -----------------------
# Import traci after adding tools path
# -----------------------
try:
    import traci
except Exception as e:
    sys.exit(f"Failed to import traci: {e}")

# -----------------------
# Sumo start configuration
# -----------------------
Sumo_config = [
    sumo_binary,
    "-c", SUMO_CONFIG_PATH,
    "--step-length", "0.10",
    "--lateral-resolution", "0"
]

# -------------------------
# RL Hyperparameters & state
# -------------------------
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.0           # as in your code (deterministic policy)
ACTIONS = [0, 1]        # 0 = keep phase, 1 = switch phase

Q_table = {}
MIN_GREEN_STEPS = 0     # as in your code
last_switch_step = -MIN_GREEN_STEPS

# Detector and TLS IDs (must match your SUMO files)
DETECTOR_IDS = {
    "Node1_2_EB_0": "Node1_2_EB_0",
    "Node1_2_EB_1": "Node1_2_EB_1",
    "Node1_2_EB_2": "Node1_2_EB_2",
    "Node2_7_SB_0": "Node2_7_SB_0",
    "Node2_7_SB_1": "Node2_7_SB_1",
    "Node2_7_SB_2": "Node2_7_SB_2",
}
TRAFFIC_LIGHT_ID = "Node2"

# -------------------------
# Safe wrappers & RL funcs
# -------------------------
def get_max_Q_value_of_state(s):
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

def get_reward(state):
    total_queue = sum(state[:-1])  # exclude current_phase
    return -float(total_queue)

def safe_get_detector_count(det_id):
    """Return last step vehicle count for detector or 0 if missing/error."""
    try:
        return int(traci.lanearea.getLastStepVehicleNumber(det_id))
    except Exception:
        return 0

def safe_get_tls_phase(tls_id):
    try:
        return int(traci.trafficlight.getPhase(tls_id))
    except Exception:
        return 0

def get_state():
    """
    Returns tuple: (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)
    Uses safe accessors to avoid exceptions if detectors are missing.
    """
    q_EB_0 = safe_get_detector_count(DETECTOR_IDS["Node1_2_EB_0"])
    q_EB_1 = safe_get_detector_count(DETECTOR_IDS["Node1_2_EB_1"])
    q_EB_2 = safe_get_detector_count(DETECTOR_IDS["Node1_2_EB_2"])
    q_SB_0 = safe_get_detector_count(DETECTOR_IDS["Node2_7_SB_0"])
    q_SB_1 = safe_get_detector_count(DETECTOR_IDS["Node2_7_SB_1"])
    q_SB_2 = safe_get_detector_count(DETECTOR_IDS["Node2_7_SB_2"])
    current_phase = safe_get_tls_phase(TRAFFIC_LIGHT_ID)
    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)

def apply_action(action, tls_id=TRAFFIC_LIGHT_ID, current_simulation_step=0):
    """
    Performs chosen action. With MIN_GREEN_STEPS guard.
    action: 0 keep, 1 switch (to next phase)
    """
    global last_switch_step
    if action == 0:
        return
    if action == 1:
        try:
            if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
                program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                num_phases = len(program.phases)
                next_phase = (safe_get_tls_phase(tls_id) + 1) % num_phases
                traci.trafficlight.setPhase(tls_id, next_phase)
                last_switch_step = current_simulation_step
        except Exception:
            # ignore action if TLS not present or call fails
            return

def update_Q_table(old_state, action, reward, new_state):
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    old_q = Q_table[old_state][action]
    best_future_q = get_max_Q_value_of_state(new_state)
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def get_action_from_policy(state):
    # YOUR original deterministic behavior: always return 0 when EPSILON=0
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        # your provided code returned 0 always; keep that behavior
        return 0

# -------------------------
# Simulation loop
# -------------------------
step_history = []
reward_history = []
queue_history = []

cumulative_reward = 0.0

try:
    print("Starting SUMO via TraCI ...")
    traci.start(Sumo_config)
    # Attempt to set GUI schema, but ignore errors (some SUMO versions/views differ)
    try:
        traci.gui.setSchema("View #0", "real world")
    except Exception:
        pass

    print("\n=== Starting Fully Online Continuous Learning ===")
    for step in range(TOTAL_STEPS):
        current_simulation_step = step

        state = get_state()
        # Deterministic policy: returns 0 (keep phase)
        action = get_action_from_policy(state)
        # apply_action is kept but since action=0, no change will be applied
        apply_action(action, current_simulation_step=current_simulation_step)

        # Advance SUMO simulation by one step
        traci.simulationStep()

        new_state = get_state()
        reward = get_reward(new_state)
        cumulative_reward += reward

        # Update Q-table if you wish to train (your provided run commented out updates).
        # Uncomment the next line if you want learning even with deterministic policy:
        # update_Q_table(state, action, reward, new_state)

        # Logging every 100 steps (as in your modified script)
        if step % 100 == 0:
            print(f"Step {step}, Current_State: {state}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")
            step_history.append(step)
            reward_history.append(cumulative_reward)
            queue_history.append(sum(new_state[:-1]))
            print("Current Q-table size:", len(Q_table))
            # optionally print Q-table contents (comment out if too verbose)
            for st, qvals in Q_table.items():
                print(f"  {st} -> {qvals}")

except Exception as e:
    print("Exception during simulation:", e)
finally:
    try:
        traci.close()
    except Exception:
        pass

# -------------------------
# Final reporting & visualization
# -------------------------
print("\nOnline run completed. Final Q-table size:", len(Q_table))
for st, actions in Q_table.items():
    print("State:", st, "-> Q-values:", actions)

# Plotting (if display available)
if len(step_history) > 0:
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
        plt.xlabel("Simulation Step")
        plt.ylabel("Cumulative Reward")
        plt.title("Fixed Timing: Cumulative Reward over Steps")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
        plt.xlabel("Simulation Step")
        plt.ylabel("Total Queue Length")
        plt.title("Fixed Timing: Queue Length over Steps")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("Plotting failed (likely no display):", e)
