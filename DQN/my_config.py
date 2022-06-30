"""
    id:  (str)
        the id of the environment, usually first 3 letters, example value = Pac
    is_atari: (bool)
        whether it is an atari-2600 game or not, value = [True, False]
    fire_reset: (bool)
        whether fire-reset is used in atari game, values = [True, False]
    max_frames: (int)
        max number of iterations in training, example value = 4000000,
    learning_rate: (float)
        learning rate for the optimizer, example value = 1e-4
    act_strategy: (str)
        the strategy used for exploration when selecting agents actions, values = [e_greedy, softmax]
    epsilon_decay: (int or float)
        the decay of epsilon value, when using e_greedy strategy, example value = 2 * 1e5
    epsilon_final: (float)
        the least value for epsilon, when using e_greedy strategy, example value = 0.1
    batch_size: (int)
        batch size used for training the agent, example value = 64
    buffer_size: (int)
        size used for experience buffer, example value = 1e5
    agent_saving_gain: (int)
        the least gain in average_rewards to achieve before saving agent, example value = 250
"""


CONFIG = {
    "CartPole-v1": {
        "id": "Cart",
        "is_atari": False,
        "fire_reset": False,
        "max_frames": 1e5,
        "learning_rate": 1e-3,
        "act_strategy": "e_greedy",
        "epsilon_decay": 2 * 1e4,
        "epsilon_final": 0.1,
        "batch_size": 32,
        "buffer_size": 10000,
        "agent_saving_gain": 30,
    },
    "PongNoFrameskip-v4": {
        "id": "Pong",
        "is_atari": True,
        "fire_reset": True,
        "max_frames": 1e6,
        "learning_rate": 1e-4,
        "act_strategy": "e_greedy",
        "epsilon_decay": 1.5 * 1e5,
        "epsilon_final": 0.01,
        "batch_size": 32,
        "buffer_size": 10000,
        "agent_saving_gain": 3,
    },
    "SpaceInvaders-v0": {
        "id": "Space",
        "is_atari": True,
        "fire_reset": True,
        "max_frames": 1e6,
        "learning_rate": 1e-4,
        "act_strategy": "e_greedy",
        "epsilon_decay": 2 * 1e5,
        "epsilon_final": 0.1,
        "batch_size": 32,
        "buffer_size": 10000,
        "agent_saving_gain": 50,
    },
    "MsPacman-v0": {
        "id": "Pac",
        "is_atari": True,
        "fire_reset": False,
        "max_frames": 4*1e6,
        "learning_rate": 1e-4,
        "act_strategy": "softmax",
        "epsilon_decay": 2 * 1e5,
        "epsilon_final": 0.1,
        "batch_size": 32,
        "buffer_size": 1e5,
        "agent_saving_gain": 250,
    }
}

def get_config(env_name):
    return CONFIG[env_name]
