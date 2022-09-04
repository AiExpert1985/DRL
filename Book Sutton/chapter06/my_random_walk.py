import numpy as np
from tqdm import tqdm

STATES = ['L_EXIT', 'A', 'B', 'C', 'D', 'E', 'R_EXIT']
INITIAL_VALUE = {state: 0.5 for state in STATES[1:6]}
FINAL_VALUE = {state: (i+1)/6 for i, state in enumerate(STATES[1:6])}

LEFT = 0
RIGHT = 1
ACTIONS = [LEFT, RIGHT]

def reward(state_a, state_b):
    if state_a == 'E' and state_b == 'R_EXIT':
        return 1
    return 0

def transition(state):
    pass
