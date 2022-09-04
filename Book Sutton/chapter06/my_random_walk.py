import numpy as np
from tqdm import tqdm

STATES = ['LEFT', 'A', 'B', 'C', 'D', 'E', 'RIGHT']
INITIAL_VALUE = {state: 0.5 for state in STATES[1:6]}
FINAL_VALUE = {state: (i+1)/6 for i, state in enumerate(STATES[1:6])}

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
    rewards = 0
    state = 'C'
    while state != 'RIGHT' and state != 'LEFT':
        next_state, reward = transition(state)
        rewards += reward
        state = next_state
    return rewards


def run():
    episodes = 1000000
    rewards = 0
    for _ in tqdm(range(episodes)):
        rewards += run_episode()
    print(f'average rewards of {episodes} episode = {round(rewards/episodes, 2)}')


def temporal_difference():
    pass


def monte_carlo():
    pass


if __name__ == '__main__':
    run()
