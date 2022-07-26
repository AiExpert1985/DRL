import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Bandit:
    def __init__(self, arms=10, epsilon=0.0, lr=0.1, q_true_initial=0., q_estimated_initial=0.,
                 is_sample_avg=False, is_ucb=False, is_grad=False, is_stationary=True):
        self.arms = arms
        self.actions = np.arange(self.arms)
        self.epsilon = epsilon
        self.lr = lr
        self.is_sample_avg = is_sample_avg
        self.is_ucb = is_ucb
        self.is_grad = is_grad
        self.is_stationary = is_stationary
        self.q_true_initial = q_true_initial
        self.q_estimated_initial = q_estimated_initial

    def reset(self):
        if self.is_stationary:
            self.q_true = np.random.randn(self.arms)
        else:
            self.q_true = np.zeros(self.arms) + self.q_true_initial
        self.q_estimated = np.zeros(self.arms) + self.q_estimated_initial
        self.action_counts = np.zeros(self.arms)
        self.t = 0

    def act(self):
        best_true_q = np.max(self.q_true)
        best_estimated_q = np.max(self.q_estimated)
        if self.is_ucb:
            action = 1
        elif self.is_grad:
            action = 1
        else:
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.actions)
            else:
                action = np.random.choice(np.where(self.q_estimated == best_estimated_q)[0])
        self.action_counts[action] += 1
        is_best_action = self.q_true[action] == best_true_q
        return action, is_best_action

    def step(self, action):
        self.t += 1
        if self.is_stationary:
            self.q_true += np.random.normal(0, 0.01)
        reward = self.q_true[action] + np.random.randn()
        self.action_counts[action] += 1
        if self.is_sample_avg:
            self.lr = 1 / self.action_counts[action]
        self.q_estimated[action] += self.lr * (reward - self.q_estimated[action])
        return reward


def run_simulation(bandits, runs, time):
    rewards = np.zeros((len(bandits), runs, time))
    best_actions = np.zeros(rewards.shape)
    for b, bandit in enumerate(bandits):
        for r in tqdm(np.arange(runs)):
            bandit.reset()
            for t in np.arange(time):
                action, is_best_action = bandit.act()
                best_actions[b, r, t] = is_best_action
                reward = bandit.step(action)
                rewards[b, r, t] = reward
    rewards = np.mean(rewards, axis=1)
    best_actions = np.mean(best_actions, axis=1)
    return rewards, best_actions


def section_2_3():
    epsilons = [0.0, 0.01, 0.1]
    bandits = [Bandit(epsilon=epsilon, is_sample_avg=True) for epsilon in epsilons]
    rewards, best_actions = run_simulation(bandits, runs=2000, time=1000)

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


def exercise_2_5():
    is_sample_average = [True, False]
    bandits = [Bandit(arms=20, epsilon=0.1, lr=0.1, is_stationary=False, is_sample_avg=is_avg)
               for is_avg in is_sample_average]
    rewards, best_actions = run_simulation(bandits, runs=1000, time=1000)

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    labels = ['sample_avg', 'fixed_step']

    for r, l in zip(rewards, labels):
        plt.plot(r, label=l)
    plt.xlabel('reward')
    plt.ylabel('time')
    plt.legend()
    plt.subplot(2, 1, 2)

    for p, l in zip(rewards, labels):
        plt.plot(p, label=l)
    plt.xlabel('% optimal action')
    plt.ylabel('time')
    plt.legend()

    plt.savefig('../images/ex_2_5')


if __name__ == '__main__':
    exercise_2_5()
