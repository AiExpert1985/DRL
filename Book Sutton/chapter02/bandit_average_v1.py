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
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.t = 0

    def act(self):
        if np.random.randn() < self.epsilon:
            return np.random.choice(np.arange(self.k))
        # if more than one state has similar value, choose one of them randomly
        q_best = np.max(self.q_estimated)
        return np.random.choice(np.where(self.q_estimated == q_best)[0])

    def step(self, action):
        self.t += 1
        self.action_count[action] += 1
        reward = np.random.randn() + self.q_true[action]
        self.q_estimated[action] += (reward - self.q_estimated[action]) / self.action_count[action]
        return reward


def run(epsilons, runs, time):
    rewards = np.zeros((len(epsilons), runs, time))
    is_best_actions = np.zeros(rewards.shape)
    for i, e in enumerate(epsilons):
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
    return reward_results, best_action_results


def fig_2_4():
    epsilons = [0.0, 0.01, 0.1]
    runs = 2000
    time = 1000
    rewards, best_actions = run(epsilons, runs, time)
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for e, rew in zip(epsilons, rewards):
        plt.plot(rew, label=e)
    plt.xlabel("time")
    plt.ylabel("rewards mean")
    plt.legend()
    plt.subplot(2, 1, 2)
    for e, a in zip(epsilons, best_actions):
        plt.plot(a, label=e)
    plt.xlabel("time")
    plt.ylabel("rewards mean")
    plt.legend()
    plt.show()
    # plt.savefig('../images/fig_2_2.png')


if __name__ == '__main__':
    fig_2_4()

