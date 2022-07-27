import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Bandit:
    def __init__(self, arms=10, epsilon=0., step_size=0.1, q_estimated_initial=0., gradient_baseline=True,
                 is_sample_avg=False, ucb_param=None, is_gradient=False, is_nonstationary=False):
        self.arms = arms
        self.actions = np.arange(self.arms)
        self.epsilon = epsilon
        self.step_size = step_size
        self.is_sample_avg = is_sample_avg
        self.ucb_param = ucb_param
        self.is_gradient = is_gradient
        self.is_nonstationary = is_nonstationary
        self.q_estimated_initial = q_estimated_initial
        self.gradient_baseline = gradient_baseline

    def reset(self):
        self.q_true = np.zeros(self.arms) if self.is_nonstationary else np.random.randn(self.arms)
        self.q_estimated = np.zeros(self.arms) + self.q_estimated_initial
        self.action_counts = np.zeros(self.arms)
        self.reward_mean = 0
        self.t = 0

    def act(self):
        best_true_q = np.max(self.q_true)
        if self.ucb_param is not None:
            ucb = self.q_estimated + \
                  self.ucb_param * np.sqrt(np.log(self.t+1)/(self.action_counts + 0.00001))
            max_val = np.max(ucb)
            action = np.random.choice(np.where(ucb == max_val)[0])
        elif self.is_gradient:
            exp_q_est = np.exp(self.q_estimated)
            self.policy = exp_q_est / np.sum(exp_q_est)
            action = np.random.choice(self.actions, p=self.policy)
        else:
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.actions)
            else:
                best_estimated_q = np.max(self.q_estimated)
                action = np.random.choice(np.where(self.q_estimated == best_estimated_q)[0])
        self.action_counts[action] += 1
        is_best_action = self.q_true[action] == best_true_q
        return action, is_best_action

    def step(self, action):
        self.t += 1
        reward = self.q_true[action] + np.random.randn()
        self.action_counts[action] += 1
        if self.is_sample_avg:
            self.step_size = 1 / self.action_counts[action]
        if self.is_gradient:
            mask = np.zeros(self.arms)
            mask[action] = 1
            if self.gradient_baseline:
                self.q_estimated += self.step_size * (reward - self.reward_mean) * (mask - self.policy)
            else:
                self.q_estimated += self.step_size * reward * (mask - self.policy)
        else:
            self.q_estimated[action] += self.step_size * (reward - self.q_estimated[action])
        if self.is_nonstationary:
            self.q_true += np.random.normal(0, 0.01, self.arms)
        self.reward_mean += (reward - self.reward_mean) / self.t
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


def section_2_3(runs=1000, time=1000):
    epsilons = [0.0, 0.01, 0.1]
    bandits = [Bandit(epsilon=epsilon, is_sample_avg=True) for epsilon in epsilons]
    rewards, best_actions = run_simulation(bandits, runs=runs, time=time)

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


def exercise_2_5(runs=1000, time=10000):
    bandits = [Bandit(epsilon=0.1, is_nonstationary=True, is_sample_avg=True, step_size=0.1),
               Bandit(epsilon=0.1, is_nonstationary=True, is_sample_avg=False)]
    rewards, best_actions = run_simulation(bandits, runs=runs, time=time)

    plt.figure(figsize=(10, 20))
    labels = ['sample_avg', 'fixed_step']

    plt.subplot(2, 1, 1)
    for r, l in zip(rewards, labels):
        plt.plot(r, label=l)
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for p, l in zip(best_actions, labels):
        plt.plot(p, label=l)
    plt.xlabel('time')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('../images/ex_2_5')


def section_2_6(runs=1000, time=1000):
    bandits = [Bandit(epsilon=0., q_estimated_initial=5.),
               Bandit(epsilon=0.1, q_estimated_initial=0.)]
    rewards, best_actions = run_simulation(bandits, runs=runs, time=time)

    plt.figure(figsize=(10, 20))
    labels = ['optimistic', 'Realistic']

    plt.subplot(2, 1, 1)
    for r, l in zip(rewards, labels):
        plt.plot(r, label=l)
        plt.xlabel('time')
        plt.ylabel('rewards')
        plt.legend()

    plt.subplot(2, 1, 2)
    for a, l in zip(best_actions, labels):
        plt.plot(a, label=l)
        plt.xlabel('time')
        plt.ylabel('% best actions')
        plt.legend()

    plt.savefig('../images/sec_2_6.png')


def section_2_7(runs=1000, time=1000):
    bandits = [Bandit(epsilon=0., is_sample_avg=True, ucb_param=2.),
               Bandit(epsilon=.1, is_sample_avg=True)]
    rewards, best_actions = run_simulation(bandits, runs=runs, time=time)

    plt.figure(figsize=(10, 20))
    labels = ['ucb', 'e-greedy']

    plt.subplot(2, 1, 1)
    for r, l in zip(rewards, labels):
        plt.plot(r, label=l)
        plt.xlabel('time')
        plt.ylabel('rewards')
        plt.legend()

    plt.subplot(2, 1, 2)
    for a, l in zip(best_actions, labels):
        plt.plot(a, label=l)
        plt.xlabel('time')
        plt.ylabel('% best actions')
        plt.legend()

    plt.savefig('../images/sec_2_7.png')


def section_2_8(runs=1000, time=1000):
    bandits = [Bandit(step_size=0.1, is_gradient=True, gradient_baseline=True),
               Bandit(step_size=0.1, is_gradient=True, gradient_baseline=False),
               Bandit(step_size=0.4, is_gradient=True, gradient_baseline=True),
               Bandit(step_size=0.4, is_gradient=True, gradient_baseline=False),
               ]
    rewards, best_actions = run_simulation(bandits, runs=runs, time=time)

    plt.figure(figsize=(10, 20))
    labels = ['alpha=0.1 with baseline',
              'alpha=0.1 without baseline',
              'alpha=0.4 with baseline',
              'alpha=0.4 without baseline', ]

    plt.subplot(2, 1, 1)
    for r, l in zip(rewards, labels):
        plt.plot(r, label=l)
        plt.xlabel('time')
        plt.ylabel('rewards')
        plt.legend()

    plt.subplot(2, 1, 2)
    for a, l in zip(best_actions, labels):
        plt.plot(a, label=l)
        plt.xlabel('time')
        plt.ylabel('% best actions')
        plt.legend()

    plt.savefig('../images/sec_2_8.png')


if __name__ == '__main__':
    section_2_8()
