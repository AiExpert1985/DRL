import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter


class Bandit:
    def __init__(self, k_arms, epsilon):
        self.k = k_arms
        self.epsilon = epsilon

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.q_estimated = np.zeros(self.k)
        self.action_counts = np.zeros(self.k)
        self.t = 0

    def act(self):
        if np.random.randn() < self.epsilon:
            return np.random.choice(np.arange(self.k))
        return np.argmax(self.q_estimated)

    def step(self, action):
        self.t += 1
        self.action_counts[action] += 1
        reward = np.random.randn() + self.q_true[action]
        self.q_estimated[action] += (reward - self.q_estimated[action]) / self.action_counts[action]
        return reward

def run():
    writer = SummaryWriter(comment='-Bandit_avg')
    k_arms = 10
    bandit_10 = Bandit(k_arms=k_arms, epsilon=0.01)
    bandit_10.reset()
    for i in range(1000000):
        action = bandit_10.act()
        bandit_10.step(action)
        diff = np.sum(np.abs(bandit_10.q_estimated-bandit_10.q_true)) / k_arms
        if i % 100000 == 0:
            print(f'{i}: {diff: .3f}')
            writer.add_scalar('diff', diff, i)

    print([(a, b) for (a, b) in zip(bandit_10.q_estimated, bandit_10.q_true)])


if __name__ == '__main__':
    run()

