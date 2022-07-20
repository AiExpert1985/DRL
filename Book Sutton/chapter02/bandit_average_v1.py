import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, k_arms, epsilon):
        self.k = k_arms
        self.epsilon = epsilon

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.q_estimated = np.zeros(self.k)
        self.action_counts = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
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
    epsilons = [0.0, 0.01, 0.1]
    runs = 2000
    time = 1000
    rewards = np.zeros((len(epsilons), runs, time))
    is_best_actions = np.zeros(rewards.shape)
    for i, e in enumerate([0.0, 0.01, 0.1]):
        bandit = Bandit(k_arms=10, epsilon=e)
        for r in tqdm(range(runs)):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    is_best_actions[i, r, t] = 1
    reward_results = np.mean(rewards, axis=1)
    best_action_results = np.mean(is_best_actions, axis=1)
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for e, rew in zip(epsilons, reward_results):
        plt.plot(rew, label=e)
    plt.xlabel("time")
    plt.ylabel("rewards mean")
    plt.legend()
    plt.subplot(2, 1, 2)
    for e, a in zip(epsilons, best_action_results):
        plt.plot(a, label=e)
    plt.xlabel("time")
    plt.ylabel("rewards mean")
    plt.legend()
    plt.show()
    # plt.savefig('../images/fig_2_2.png')


if __name__ == '__main__':
    run()

