import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use('Agg')


class Bandit:
    def __init__(self, k_arms, epsilon):
        self.k = k_arms
        self.epsilon = epsilon

    def reset(self):
        self.q_true = np.ones(self.k)  # all q_true values are equal at the beginning
        self.q_estimated = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.t = 0

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.k))
        # if more than one state has similar value, choose one of them randomly
        q_best = np.max(self.q_estimated)
        return np.random.choice(np.where(self.q_estimated == q_best)[0])

    def step(self, action, alpha_type):
        self.t += 1
        self.action_count[action] += 1
        reward = np.random.randn() + self.q_true[action]
        alpha = 0.1 if alpha_type == "constant" else 1 / self.action_count[action]
        self.q_estimated[action] += alpha * (reward - self.q_estimated[action])
        self.q_true += np.random.normal(0, 0.01, self.k)  # q_true take random walk
        # self.q_true += np.random.randn(self.k)
        self.best_action = np.argmax(self.q_true)
        return reward


def simulate(bandit, runs, time, alpha_types):
    rewards = np.zeros((len(alpha_types), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, alpha_type in enumerate(alpha_types):
        for r in tqdm(range(runs)):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action, alpha_type)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_rewards = np.mean(rewards, axis=1)
    mean_best_action_counts = np.mean(best_action_counts, axis=1)
    return mean_best_action_counts, mean_rewards


def ex_2_5():
    alpha_types = ["constant", "variable"]
    bandit = Bandit(k_arms=10, epsilon=0.1)
    best_action_counts, rewards = simulate(bandit, runs=2000, time=10000, alpha_types=alpha_types)
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for e, rew in zip(alpha_types, rewards):
        plt.plot(rew, label=e)
    plt.xlabel("time")
    plt.ylabel("rewards mean")
    plt.legend()
    plt.subplot(2, 1, 2)
    for e, a in zip(alpha_types, best_action_counts):
        plt.plot(a, label=e)
    plt.xlabel("time")
    plt.ylabel("% optimal action")
    plt.legend()
    plt.savefig('../images/exercise_2_5_(2).png')


if __name__ == '__main__':
    ex_2_5()

