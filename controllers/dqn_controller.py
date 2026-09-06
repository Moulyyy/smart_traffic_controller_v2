"""
DEEP Q-NETWORK (DQN) TRAFFIC SIGNAL CONTROLLER

Key Characteristics (per project presentation Slides 10 & 13):
- Algorithm: Deep Q-Network (DQN) Reinforcement Learning.
- Neural network replaces Q-table for estimating action-values across state spaces.
- State Space (7 values):
    - Vehicle counts from 6 lane detectors (Node1_2_EB_0..2, Node2_7_SB_0..2)
    - Current traffic light phase
- Action Space (2 actions):
    - 0: Keep phase unchanged
    - 1: Switch to next phase (subject to MIN_GREEN_STEPS = 5 safety guard)
- Reward Function:
    - R = -(sum of queue lengths)
- Neural Network Architecture:
    - Input: 7 features
    - Dense layer 1: 24 units, ReLU activation
    - Dense layer 2: 24 units, ReLU activation
    - Output layer: 2 units (linear activation for Q(s, keep) and Q(s, switch))
    - Optimizer: Adam (learning_rate=0.001), Loss: Mean Squared Error (MSE)
- Hyperparameters:
    - ALPHA: 0.1
    - GAMMA: 0.9
    - EPSILON: 0.1
    - MIN_GREEN_STEPS: 5 seconds

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
import random
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
# Simulation Settings & Hyperparameters
# ---------------------------------------------------------
STEP_LENGTH = 1.0       # seconds per simulation step
TOTAL_STEPS = 3000      # total simulation steps

ALPHA = 0.1             # Learning rate
GAMMA = 0.9             # Discount factor
EPSILON = 0.1           # Exploration rate
ACTIONS = [0, 1]        # 0 = keep phase, 1 = switch phase
MIN_GREEN_STEPS = 5     # Minimum green duration guard in seconds

DETECTORS = [
    "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2",
    "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"
]
TRAFFIC_LIGHT_ID = "Node2"
last_switch_step = -MIN_GREEN_STEPS


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


def build_dqn_model(state_size=7, action_size=2):
    """Construct DQN neural network as defined in presentation Slide 13."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Suppress verbose TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(24, activation='relu', name='dense_1'),
        layers.Dense(24, activation='relu', name='dense_2'),
        layers.Dense(action_size, activation='linear', name='q_values')
    ], name="DQN_Traffic_Controller")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    return model


def to_array(state_tuple):
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))


def get_state(traci_instance):
    """Construct 7-dimensional state vector."""
    q_vals = [int(traci_instance.lanearea.getLastStepVehicleNumber(det)) for det in DETECTORS]
    phase = int(traci_instance.trafficlight.getPhase(TRAFFIC_LIGHT_ID))
    return (*q_vals, phase)


def get_reward(state):
    """R = -(sum of queue lengths)"""
    total_queue = sum(state[:-1])
    return -float(total_queue)


def get_action_from_policy(state, model):
    """Epsilon-greedy action selection using DQN network."""
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    q_values = model.predict(to_array(state), verbose=0)[0]
    return int(np.argmax(q_values))


def apply_action(action, traci_instance, current_step):
    """Switch phase if action=1 and MIN_GREEN_STEPS have elapsed."""
    global last_switch_step
    if action == 0:
        return
    if current_step - last_switch_step >= MIN_GREEN_STEPS:
        try:
            program = traci_instance.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)[0]
            num_phases = len(program.phases)
            next_phase = (int(traci_instance.trafficlight.getPhase(TRAFFIC_LIGHT_ID)) + 1) % num_phases
            traci_instance.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_phase)
            last_switch_step = current_step
        except Exception:
            pass


def update_dqn(model, old_state, action, reward, new_state):
    """DQN Bellman target update and backpropagation step."""
    old_array = to_array(old_state)
    new_array = to_array(new_state)

    q_old = model.predict(old_array, verbose=0)[0]
    q_new = model.predict(new_array, verbose=0)[0]

    best_future_q = np.max(q_new)
    q_old[action] = q_old[action] + ALPHA * (reward + GAMMA * best_future_q - q_old[action])
    model.fit(old_array, np.array([q_old]), verbose=0)


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
    """Execute Deep Q-Network simulation and online training loop."""
    global last_switch_step
    last_switch_step = -MIN_GREEN_STEPS

    sumo_binary = setup_sumo_environment()
    import traci

    # Initialize neural network
    model = build_dqn_model(state_size=7, action_size=2)
    print("\nNeural Network Architecture (Slide 13):")
    model.summary()

    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CONFIG_PATH,
        "--step-length", str(STEP_LENGTH),
        "--lateral-resolution", "0"
    ]

    print("\n========================================================")
    print("🚦 STARTING SUMO: DEEP Q-NETWORK (DQN) CONTROLLER")
    print("========================================================")
    print(f"SUMO config: {SUMO_CONFIG_PATH}")
    print(f"Hyperparameters: ALPHA={ALPHA}, GAMMA={GAMMA}, EPSILON={EPSILON}, MIN_GREEN={MIN_GREEN_STEPS}s")
    print("--------------------------------------------------------\n")

    step_history = []
    time_history = []
    queue_history = []
    reward_history = []
    cumulative_reward = 0.0
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

            # 1. Observe state
            state = get_state(traci)

            # 2. Select action via epsilon-greedy with DQN
            action = get_action_from_policy(state, model)

            # 3. Apply action
            apply_action(action, traci, step)

            # 4. Step simulation
            traci.simulationStep()

            # 5. Track vehicle trips
            for vid in traci.simulation.getDepartedIDList():
                depart_times[vid] = current_time

            for vid in traci.simulation.getArrivedIDList():
                dt = current_time - depart_times.pop(vid, current_time)
                travel_times.append(dt)

            # 6. Observe next state & reward
            new_state = get_state(traci)
            reward = get_reward(new_state)
            cumulative_reward += reward

            # 7. Update DQN Network
            update_dqn(model, state, action, reward, new_state)

            current_queue = sum(new_state[:-1])
            queue_history.append(current_queue)
            step_history.append(step)
            time_history.append(current_time)
            reward_history.append(cumulative_reward)

            if step % 200 == 0:
                q_vals = model.predict(to_array(state), verbose=0)[0]
                print(f"Step {step:4d} | Time={current_time:6.1f}s | Action={action} | "
                      f"Reward={reward:5.1f} | CumReward={cumulative_reward:8.1f} | "
                      f"Q-values={np.round(q_vals, 2)} | Queue={current_queue:2d} veh")

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
        print("=== DQN CONTROLLER METRICS ===")
        print("=" * 36)
        print(f"Total delay (veh-seconds):      {total_delay:.2f}")
        print(f"Average queue length (vehicles): {avg_queue:.2f}")
        print(f"Max queue length (vehicles):     {max_queue}")
        print(f"Vehicles completed (throughput): {throughput}")
        print(f"Average travel time / veh (s):   {avg_tt:.2f}")
        print(f"90th percentile travel time (s): {p90_tt:.2f}")
        print(f"Max travel time (s):             {max_tt:.2f}")
        print("=" * 36 + "\n")

        # Save results
        npz_path = os.path.join(OUTPUTS_DIR, "results_dqn.npz")
        np.savez(
            npz_path,
            time=np.array(time_history, dtype=np.float32),
            queue=np.array(queue_history, dtype=np.float32),
            reward=np.array(reward_history, dtype=np.float32),
            travel_times=np.array(travel_times, dtype=np.float32)
        )
        print(f"📁 Results saved to: {npz_path}")

        # Plots
        plt.figure(figsize=(10, 5))
        plt.plot(time_history, queue_history, color='tab:purple', label="DQN Queue Length")
        plt.xlabel("Time (s)")
        plt.ylabel("Total Queue Length (vehicles)")
        plt.title("DQN: Queue Length Over Time")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plot_path = os.path.join(OUTPUTS_DIR, "dqn_queue_plot.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"📊 Queue plot saved to: {plot_path}")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(time_history, reward_history, color='tab:red', label="Cumulative Reward")
        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative Reward")
        plt.title("DQN: Cumulative Reward Over Time")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        reward_plot_path = os.path.join(OUTPUTS_DIR, "dqn_reward_plot.png")
        plt.savefig(reward_plot_path, dpi=200, bbox_inches='tight')
        print(f"📊 Reward plot saved to: {reward_plot_path}")
        plt.show()


if __name__ == "__main__":
    run()
