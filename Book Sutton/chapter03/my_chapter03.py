import numpy as np
import matplotlib.pyplot as plt
import matplotlib


WORLD_SIZE = 5
MAX_ITERATIONS = 100000

A = (0, 1)
A_PRIME = (4, 1)
B = (0, 3)
B_PRIME = (2, 3)

DISCOUNT = 0.9

ACTIONS = {
    (0, 1): '↑',
    (-1, 0): '←',
    (0, -1): '↓',
    (1, 0): '→',
}

def policy_uniform(action):
    return 0.25


def inside_grid(state):
    x, y = state
    return 0 <= x < WORLD_SIZE and 0 <= y < WORLD_SIZE


def step(state, action):
    if state == A:
        return A_PRIME, +10.0
    if state == B:
        return B_PRIME, +5.0
    new_state = (state[0] + action[0], state[1] + action[1])
    if inside_grid(new_state):
        return new_state, 0.0
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
    ax.add_table(table)
    plt.show()


def fig_3_2():
    vals = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i in range(MAX_ITERATIONS):
        new_vals = np.zeros_like(vals)
        for x in range(WORLD_SIZE):
            for y in range(WORLD_SIZE):
                state = (x, y)
                for a in ACTIONS.keys():
                    (x_new, y_new), reward = step(state, a)
                    prob_a = policy_uniform(a)
                    new_vals[x, y] += prob_a * (reward + DISCOUNT * vals[x_new, y_new])
        if np.abs(np.sum(new_vals - vals)) < 0.0001:
            print(f'converged in {i} iterations')
            break
        vals = new_vals
    draw_grid(vals)


def fig_3_5():
    pass


if __name__ == '__main__':
    # fig_3_2()
    fig_3_5()
