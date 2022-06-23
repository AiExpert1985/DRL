import gym


def get_make_env_fn(**kargs):
    def make_env_fn(env_name, render=None):
        env = None
        if render:
            try:
                env = gym.make(env_name, render=render)
            except:
                pass
        if env is None:
            env = gym.make(env_name)
        return env
    return make_env_fn, kargs


class DQN:
    def __init__(self):
        pass

    def train(self):
        print('hi')

if __name__ == "__main__":
    env_settings = {
        'env_name': 'CartPole-v1',
        'gamma': 1.00,
        'max_episodes': 10000,
        'goal_mean_100_reward': 475
    }
    env_name, gamma, max_episodes, goal_mean_100_reward = env_settings.values()
    make_env_fn, kargs = get_make_env_fn(env_name=env_name)
    agent = DQN()
    agent.train()
