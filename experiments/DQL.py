"""
Robust Deep Q-Network (DQN) + SUMO + TraCI Traffic Control Script

🔧 Instructions:
1. Set SUMO_CONFIG_PATH to your absolute path of RL.sumocfg
2. Ensure SUMO_HOME is added to system environment variables or define it below.
3. Verify detector IDs and traffic light ID match your SUMO network.

This version includes:
 - Path validation for SUMO & RL.sumocfg
 - Safe detector and TLS access (no crashes if missing)
 - Continuous online DQN training
 - Automatic model summary and performance plots
"""

# -------------------------
# Step 1: Imports
# -------------------------
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------
# Step 2: SUMO Setup
# -------------------------
SUMO_CONFIG_PATH = r"C:\tracii\RL.sumocfg"   # <-- set absolute path here

if 'SUMO_HOME' not in os.environ:
    sys.exit("ERROR: Please declare environment variable 'SUMO_HOME' before running this script.")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if not os.path.isdir(tools):
    sys.exit(f"ERROR: SUMO tools not found at {tools}")
sys.path.append(tools)

# -------------------------
# Step 3: Import TraCI and check SUMO binary
# -------------------------
if os.name == 'nt':
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo.exe')
else:
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')

if not os.path.isfile(sumo_binary):
    sys.exit(f"ERROR: SUMO binary not found at {sumo_binary}")

if not os.path.isfile(SUMO_CONFIG_PATH):
    sys.exit(f"ERROR: SUMO config file not found: {SUMO_CONFIG_PATH}")

import traci

# -------------------------
# Step 4: SUMO configuration
# -------------------------
Sumo_config = [
    sumo_binary,
    "-c", SUMO_CONFIG_PATH,
    "--step-length", "0.10",
    "--lateral-resolution", "0"
]

# -------------------------
# Step 5: RL Variables and Hyperparameters
# -------------------------
TOTAL_STEPS = 10000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]
MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# -------------------------
# Step 6: Build DQN Model
# -------------------------
def build_model(state_size, action_size):
    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    return model

def to_array(state_tuple):
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))

state_size = 7  # q_EB_0..q_SB_2 + current_phase
action_size = len(ACTIONS)
dqn_model = build_model(state_size, action_size)

# -------------------------
# Step 7: Utility & RL Functions
# -------------------------
def safe_get_detector(det_id):
    try:
        return traci.lanearea.getLastStepVehicleNumber(det_id)
    except Exception:
        return 0

def safe_get_phase(tls_id):
    try:
        return traci.trafficlight.getPhase(tls_id)
    except Exception:
        return 0

def get_state():
    detectors = [
        "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2",
        "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"
    ]
    traffic_light_id = "Node2"
    q_vals = [safe_get_detector(d) for d in detectors]
    phase = safe_get_phase(traffic_light_id)
    return (*q_vals, phase)

def get_reward(state):
    total_queue = sum(state[:-1])
    return -float(total_queue)

def get_action_from_policy(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        Q_values = dqn_model.predict(to_array(state), verbose=0)[0]
        return int(np.argmax(Q_values))

def apply_action(action, tls_id="Node2", current_simulation_step=0):
    global last_switch_step
    if action == 0:
        return
    try:
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (safe_get_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step = current_simulation_step
    except Exception:
        pass

def update_DQN(old_state, action, reward, new_state):
    old_array = to_array(old_state)
    new_array = to_array(new_state)
    Q_old = dqn_model.predict(old_array, verbose=0)[0]
    Q_new = dqn_model.predict(new_array, verbose=0)[0]
    best_future_q = np.max(Q_new)
    Q_old[action] = Q_old[action] + ALPHA * (reward + GAMMA * best_future_q - Q_old[action])
    dqn_model.fit(old_array, np.array([Q_old]), verbose=0)

# -------------------------
# Step 8: Simulation Loop
# -------------------------
step_history = []
reward_history = []
queue_history = []
cumulative_reward = 0.0

print("\n=== Starting SUMO and DQN Training ===")
try:
    traci.start(Sumo_config)
    for step in range(TOTAL_STEPS):
        current_simulation_step = step
        state = get_state()
        action = get_action_from_policy(state)
        apply_action(action, current_simulation_step=current_simulation_step)
        traci.simulationStep()

        new_state = get_state()
        reward = get_reward(new_state)
        cumulative_reward += reward

        update_DQN(state, action, reward, new_state)

        if step % 100 == 0:
            Q_vals = dqn_model.predict(to_array(state), verbose=0)[0]
            print(f"Step {step}: State={state}, Action={action}, Reward={reward:.2f}, Cum.Reward={cumulative_reward:.2f}, Q={Q_vals}")
            step_history.append(step)
            reward_history.append(cumulative_reward)
            queue_history.append(sum(new_state[:-1]))

except Exception as e:
    print("Simulation Error:", e)

finally:
    traci.close()

# -------------------------
# Step 9: Results
# -------------------------
print("\nTraining Completed. DQN Model Summary:\n")
dqn_model.summary()

plt.figure(figsize=(10, 5))
plt.plot(step_history, reward_history, label="Cumulative Reward", marker='o')
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("DQN Training: Reward Progress")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(step_history, queue_history, label="Total Queue", marker='o')
plt.xlabel("Step")
plt.ylabel("Total Queue Length")
plt.title("DQN Training: Queue Length Over Steps")
plt.grid(True)
plt.legend()
plt.show()
