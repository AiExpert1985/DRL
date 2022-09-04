import numpy as np
from tqdm import tqdm
from collections import defaultdict

STATES = ['LEFT', 'A', 'B', 'C', 'D', 'E', 'RIGHT']
TRUE_VALUE = {state: (i+1)/6 for i, state in enumerate(STATES[1:6])}
TERMINAL_STATE = ['LEFT', 'RIGHT']

LEFT = -1
RIGHT = 1
ACTIONS = [LEFT, RIGHT]

ALPHA = 0.1
GAMMA = 1.0

V = defaultdict(lambda: 0.5)


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


def temporal_difference(trajectory):
    trajectory.reverse()
    for state, next_state, reward in trajectory:
        target = reward if next_state in TERMINAL_STATE else reward + GAMMA * V[next_state]
        # print(state, next_state, reward)
        # print('target =', target)
        td_error = target - V[state]
        # print('td_error =', td_error)
        # print('before V[s] =', V[state])
        V[state] += ALPHA * td_error
        # print('after V[s] =', V[state])
        # print("------------")

def rmse_error():
    se = []
    for state in STATES[1: 6]:
        e = V[state] - TRUE_VALUE[state]
        se.append(np.square(e))
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse


def monte_carlo(trajectory):
    pass


def run():
    episodes = 100
    rewards = 0
    for _ in tqdm(range(episodes)):
        trajectory, reward = run_episode()
        rewards += reward
        temporal_difference(trajectory)
        # monte_carlo(trajectory)
    for state, val in V.items():
        print(state, round(val, 2), round(TRUE_VALUE[state], 2))
    error = rmse_error()
    print('error =', round(error, 2))
    # print(f'average rewards of {episodes} episode = {round(rewards/episodes, 2)}')


if __name__ == '__main__':
    run()
