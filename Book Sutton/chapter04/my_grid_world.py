import numpy as np
import matplotlib.pyplot as plt
import matplotlib


WORLD_SIZE = 4
MAX_ITERATIONS = 1000

DISCOUNT = 1.0

ACTIONS = [(0, 1), (-1, 0), (0, -1), (1, 0)]

def inside_grid(state):
    x, y = state
    return 0 <= x < WORLD_SIZE and 0 <= y < WORLD_SIZE

def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)


def step(state, action_index):
    action = ACTIONS[action_index]
    if is_terminal(state):
        return state, 0
    new_state = (state[0] + action[0], state[1] + action[1])
    if inside_grid(new_state):
        return new_state, -1.0
    return state, -1.0


def draw_grid(vals):
    vals = np.round(vals, decimals=2)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = matplotlib.table.Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = vals.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    for (i, j), val in np.ndenumerate(vals):
        table.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')
    for i in range(len(vals)):
        table.add_cell(i, -1, width, height, text=i+1, loc='right',
                       edgecolor='none', facecolor='none')
        table.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                       edgecolor='none', facecolor='none')
    ax.add_table(table)
    plt.show()


def policy_evaluation(policy):
    V_old = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i in range(MAX_ITERATIONS):
        V = V_old.copy()
        for x in range(WORLD_SIZE):
            for y in range(WORLD_SIZE):
                state = (x, y)
                action = policy[state]
                new_state, reward = step(state, action)
                V[state] = reward + DISCOUNT * V[new_state]
        if abs(V - V_old).max() < 0.0001:
            return V
        V_old = V
    return V_old

def policy_improvement(policy, V):
    for x in range(WORLD_SIZE):
        for y in range(WORLD_SIZE):
            state = (x, y)
            action_values = []
            for action, _ in enumerate(ACTIONS):
                (x_new, y_new), reward = step(state, action)
                action_values.append(reward + DISCOUNT * V[x_new, y_new])
            policy[state] = np.argmax(action_values)
    return policy


def policy_iteration():
    policy = np.random.choice(len(ACTIONS), (WORLD_SIZE, WORLD_SIZE))
    V_old = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i in range(1000):
        V = policy_evaluation(policy)
        policy = policy_improvement(policy, V)
        if np.array_equal(V, V_old):
            print(f"Algorithm converged in {i} iterations")
            break
        V_old = V
        draw_grid(np.round(V_old, decimals=2))

def value_iteration():
    V_old = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i in range(MAX_ITERATIONS):
        V = np.zeros_like(V_old)
        for x in range(WORLD_SIZE):
            for y in range(WORLD_SIZE):
                state = (x, y)
                action_values = []
                for action, _ in enumerate(ACTIONS):
                    (x_new, y_new), reward = step(state, action)
                    action_values.append(reward + DISCOUNT * V_old[x_new, y_new])
                V[state] = np.max(action_values)
        draw_grid(np.round(V_old, decimals=2))
        if abs(V - V_old).max() < 0.0001:
            return V
        V_old = V
    return V_old


if __name__ == '__main__':
    # policy_iteration()
    value_iteration()
