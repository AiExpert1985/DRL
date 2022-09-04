import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

STATES = ['LEFT', 'A', 'B', 'C', 'D', 'E', 'RIGHT']
TRUE_VALUE = {state: (i+1)/6 for i, state in enumerate(STATES[1:6])}
TERMINAL_STATE = ['LEFT', 'RIGHT']

LEFT = -1
RIGHT = 1
ACTIONS = [LEFT, RIGHT]

ALPHA = 0.04
GAMMA = 1.0


def reward_fn(state_a, state_b):
    if state_a == 'E' and state_b == 'RIGHT':
        return 1
    return 0


def transition(state):
    action = np.random.choice(ACTIONS)
    idx = STATES.index(state)
    next_state = STATES[idx + action]
    reward = reward_fn(state, next_state)
    return next_state, reward


def run_episode():
    rewards = 0
    state = 'C'
    trajectory = []
    while state != 'RIGHT' and state != 'LEFT':
        next_state, reward = transition(state)
        rewards += reward
        trajectory.append((state, next_state, reward))
        state = next_state
    return trajectory, rewards


def temporal_difference(trajectory, V):
    trajectory.reverse()
    for state, next_state, reward in trajectory:
        target = reward if next_state in TERMINAL_STATE else reward + GAMMA * V[next_state]
        td_error = target - V[state]
        V[state] = V[state] + ALPHA * td_error


def monte_carlo(trajectory, V):
    trajectory.reverse()
    target = 0  # another name for the return (G)
    for state, _, reward in trajectory:
        target += GAMMA * reward
        mc_error = target - V[state]
        V[state] = V[state] + ALPHA * mc_error


def rmse_error(V):
    se = []
    for state in STATES[1: 6]:
        e = V[state] - TRUE_VALUE[state]
        se.append(np.square(e))
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse


def run(method=temporal_difference, episodes=100):
    V = defaultdict(lambda: 0.5)
    rewards = 0
    for _ in range(episodes):
        trajectory, reward = run_episode()
        rewards += reward
        method(trajectory, V)
    return V


def plot_6_2_left():
    true_vals = [TRUE_VALUE[state] for state in STATES[1:6]]
    for n in [1, 10, 100]:
        V = run(method=temporal_difference, episodes=n)
        vals = [V[state] for state in STATES[1:6]]
        plt.plot(STATES[1:6], vals, label=n)
    plt.plot(STATES[1:6], true_vals, label='true')
    plt.show()


def simulate():
    num_tries = 100
    errors = []
    for _ in tqdm(range(num_tries)):
        V = run()
        errors.append(rmse_error(V))
    average_error = np.mean(errors)
    print('error =', np.round(average_error, 3))
    plot_6_2_left()

if __name__ == '__main__':
    simulate()
