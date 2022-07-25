import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use('Agg')


# noinspection PyAttributeOutsideInit
class Bandit:
    def __init__(self, k_arms, alpha):
        self.k = k_arms
        self.actions = np.arange(self.k)
        self.alpha = alpha

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.q_estimated = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.preferences = np.zeros(self.k)
        self.t = 0
        self.reward_average = 0

    def act(self):
        pref_exp = np.exp(self.preferences)
        self.policy = pref_exp/np.sum(pref_exp)
        return np.random.choice(self.actions, p=self.policy)

    def step(self, action):
        self.t += 1
        self.action_count[action] += 1
        reward = np.random.randn() + self.q_true[action]
        mask = np.zeros(self.k)
        mask[action] = 1
        self.preferences = self.preferences + self.alpha * (reward - self.reward_average) * (mask-self.policy)
        self.reward_average += (1/self.t) * (reward - self.reward_average)
        return reward


def simulate(bandits, runs, time):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in tqdm(range(runs)):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_rewards = np.mean(rewards, axis=1)
    mean_best_action_counts = np.mean(best_action_counts, axis=1)
    return mean_best_action_counts, mean_rewards


def fig_2_2():
    alphas = [0.1, 0.4]
    bandits = [Bandit(k_arms=10, alpha=a) for a in alphas]
    best_action_counts, rewards = simulate(bandits, runs=2000, time=1000)
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for e, rew in zip(alphas, rewards):
        plt.plot(rew, label=e)
    plt.xlabel("time")
    plt.ylabel("rewards mean")
    plt.legend()
    plt.subplot(2, 1, 2)
    for e, a in zip(alphas, best_action_counts):
        plt.plot(a, label=e)
    plt.xlabel("time")
    plt.ylabel("% optimal action")
    plt.legend()
    plt.savefig('../images/fig_2_8.png')


if __name__ == '__main__':
    fig_2_2()
