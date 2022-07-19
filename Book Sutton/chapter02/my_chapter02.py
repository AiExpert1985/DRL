import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use('Agg')


class Bandit:
    def __init__(self, k_arm=10, epsilon=0.0, sample_averages=False):
        self.k = k_arm
        self.epsilon = epsilon
        self.indices = np.arange(self.k)
        self.average_reward = 0
        self.sample_averages = sample_averages

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.time = 0

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)
        q_best = np.max(self.q_estimation)
        best_actions = np.where(q_best == self.q_estimation)[0]
        return np.random.choice(best_actions)

    def step(self, action):
        self.time += 1
        reward = np.random.randn() + self.q_true[action]
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time
        if self.sample_averages:
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        return reward


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in tqdm(np.arange(runs)):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
        mean_best_action_counts = best_action_counts.mean(axis=1)
        mean_rewards = rewards.mean(axis=1)
        return mean_best_action_counts, mean_rewards


def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel('Action')
    plt.ylabel('Reward distribution')
    plt.savefig('../images/figure_2_1.png')
    plt.close()


def figure_2_2(runs=2000, time=1000):
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('../images/figure_2_2.png')
    plt.close()


if __name__ == '__main__':
    figure_2_1()
