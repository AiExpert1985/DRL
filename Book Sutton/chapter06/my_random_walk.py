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
    state = 'C'
    trajectory = []
    while state != 'RIGHT' and state != 'LEFT':
        next_state, reward = transition(state)
        trajectory.append((state, next_state, reward))
        state = next_state
    return trajectory


def td(trajectory, alpha, gamma, V):
    trajectory.reverse()
    for state, next_state, reward in trajectory:
        target = reward if next_state in TERMINAL_STATE else reward + gamma * V[next_state]
        td_error = target - V[state]
        V[state] = V[state] + alpha * td_error
    return V


def mc(trajectory, alpha, gamma, V):
    trajectory.reverse()
    target = 0  # target is another name for the return (G)
    for state, _, reward in trajectory:
        target = reward + gamma * target
        mc_error = target - V[state]
        V[state] = V[state] + alpha * mc_error
    return V


def rmse_error(V):
    se = []
    for state in STATES[1: 6]:
        e = V[state] - TRUE_VALUE[state]
        se.append(np.square(e))
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse


def run(method=td, episodes=100, alpha=0.05, gamma=1.0):
    V = defaultdict(lambda: 0.5)
    for _ in range(episodes):
        trajectory = run_episode()
        V = method(trajectory, alpha, gamma, V)
    error = rmse_error(V)
    return V, error


def batch_update(trajectory, method, gamma, errors, V):
    trajectory.reverse()
    target = 0
    for state, next_state, reward in trajectory:
        if method == mc:
            target = reward + gamma * target
        else:
            target = reward + gamma * V[next_state] if next_state not in TERMINAL_STATE else reward
        error = target - V[state]
        errors[state].append(error)
    return errors


def batch(episodes=100, alpha=0.05, gamma=1.0):
    for method in [td, mc]:
        trajectories = []
        V = defaultdict(lambda: 0.5)
        for _ in range(episodes):
            trajectories.append(run_episode())
            errors = defaultdict(list)
            for trajectory in trajectories:
                errors = batch_update(trajectory, method, gamma, errors, V)
            for state in errors.keys():
                V[state] = V[state] + alpha * np.mean(errors[state])
        print(f'error for method {method.__name__} = {round(rmse_error(V), 3)}')
        vals = [V[state] for state in STATES[1:6]]
        plt.plot(STATES[1:6], vals, label=f'{method.__name__}')
    true_vals = [TRUE_VALUE[state] for state in STATES[1:6]]
    plt.plot(STATES[1:6], true_vals, label='true')
    plt.legend()
    plt.show()


def plot_6_2_left():
    true_vals = [TRUE_VALUE[state] for state in STATES[1:6]]
    for n in tqdm([1, 10, 100]):
        V, _ = run(method=td, episodes=n)
        vals = [V[state] for state in STATES[1:6]]
        plt.plot(STATES[1:6], vals, label=n)
    plt.plot(STATES[1:6], true_vals, label='true')
    plt.legend()
    plt.show()


def plot_6_2_right():
    n_tries = 100
    n_episodes = [25, 50, 75, 100]
    for method in tqdm([mc, td]):
        alphas = [0.01, 0.03, 0.05]
        for alpha in alphas:
            plot_errors = []
            for n in n_episodes:
                errors = []
                for i in range(n_tries):
                    _, e = run(method=method, episodes=n, alpha=alpha)
                    errors.append(e)
                plot_errors.append(np.average(errors))
            plt.plot(n_episodes, plot_errors, label=f'{method.__name__}: {alpha}')
    plt.legend()
    plt.show()


def simulate(method=td, n_episodes=100):
    n_episodes = n_episodes
    V, error = run(method, n_episodes)
    print('error =', np.round(error, 3))
    return error


if __name__ == '__main__':
    # error = simulate()
    # plot_6_2_left()
    # plot_6_2_right()
    batch()
