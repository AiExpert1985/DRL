import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Bandit:
    def __init__(self, arms=10, epsilon=0.0, lr=0.1, is_sample_avg=False, is_ucb=False,
                 is_grad=False, is_optim=False, is_station=True):
        self.arms = arms
        self.actions = np.arange(self.arms)
        self.epsilon = epsilon
        self.lr = lr
        self.is_sample_avg = is_sample_avg
        self.is_ucb = is_ucb
        self.is_grad = is_grad
        self.is_optim = is_optim
        self.is_station = is_station

    def reset(self):
        self.t = 0
        self.q_true = np.random.randn(self.arms)
        self.q_estimated = np.zeros(self.arms)
        self.action_counts = np.zeros(self.arms)

    def act(self):
        best_true_q = np.max(self.q_true)
        best_estimated_q = np.max(self.q_estimated)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = np.random.choice(np.where(self.q_estimated == best_estimated_q)[0])
        self.action_counts[action] += 1
        is_best_action = self.q_true[action] == best_true_q
        return action, is_best_action

    def step(self, action):
        self.t += 1
        reward = self.q_true[action] + np.random.randn()
        self.action_counts[action] += 1
        if self.is_sample_avg:
            self.lr = 1 / self.t
        self.q_estimated[action] += self.lr * (reward - self.q_estimated[action])
        return reward


def run_simulation(bandits, runs=2000, time=1000):
    rewards = np.zeros((len(bandits), runs, time))
    best_actions = np.zeros(rewards.shape)
    for b, bandit in enumerate(bandits):
        for r in tqdm(np.arange(runs)):
            bandit.reset()
            for t in np.arange(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[b, r, t] = reward
                if action == np.argmax(bandit.q_true):
                    best_actions[b, r, t] = 1
    rewards = np.mean(rewards, axis=1)
    best_actions = np.mean(best_actions, axis=1)
    return rewards, best_actions


def sec_2_3():
    epsilons = [0.0, 0.01, 0.1]
    bandits = [Bandit(epsilon=epsilon, is_sample_avg=True) for epsilon in epsilons]
    rewards, best_actions = run_simulation(bandits)
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for epsilon, reward in zip(epsilons, rewards):
        plt.plot(reward, label=epsilon)
        plt.xlabel('time')
        plt.ylabel('rewards')
        plt.legend()
    plt.subplot(2, 1, 2)
    for epsilon, best_action in zip(epsilons, best_actions):
        plt.plot(best_action, label=epsilon)
        plt.xlabel('time')
        plt.ylabel('% best actions')
        plt.legend()
    plt.savefig('../images/sec_2_3.png')


if __name__ == '__main__':
    sec_2_3()
