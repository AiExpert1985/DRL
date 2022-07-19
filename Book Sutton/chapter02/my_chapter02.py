import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')


class Bandit:
    def __init__(self, epsilon, sample_averages):
        pass


def simulate(runs, time, bandits):
    pass


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


if __name__ == '__main__':
    figure_2_1()
