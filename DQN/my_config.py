"""
id:  (str) the id of the environment, usually first 3 letters, example value = Pac
rewards_mean_length: (int) the number of rewards to use for running average, example value = 100
is_atari: (bool) whether it is an atari-2600 game or not, value = [True, False]
fire_reset: (bool) whether fire-reset is used in atari game, values = [True, False]
max_frames: (int) max number of iterations in training, example value = 4000000,
learning_rate: (float) learning rate for the optimizer, example value = 1e-4
act_strategy: (str) the strategy used for exploration when selecting agents actions, values = [e_greedy, softmax]
epsilon_decay: (int or float) the decay of epsilon value, when using e_greedy strategy, example value = 2 * 1e5
epsilon_final: (float) the least value for epsilon, when using e_greedy strategy, example value = 0.1
batch_size: (int) batch size used for training the agent, example value = 64
buffer_size: (int) size used for experience buffer, example value = 1e5
use_lag_agent: (bool) whether or not using target network, values = [True, False]
lag_update_freq: (int) how often to update target network (when using target network), example value = 10000
save_trained_agent: (bool) whether or not saving the trained model during training, values = [True, False]
agent_saving_gain: (int) the least gain in average_rewards to achieve before saving agent, example value = 250
agent_load_score: (int) the value of the saved model to be loaded in test or resume mode, example value = 3417,
test_n_games: (int) for how many episodes to run the test mode, example value = 10
with_graphics: (bool) whether or not to display the the game for human, values = [True, False]
force_cpu: (bool) force using the CPU even if PC have a GPU, values = [True, False]
"""


CONFIG = {
    "CartPole-v1": {
        "id": "Cart",
        "rewards_mean_length": 100,
        "is_atari": False,
        "fire_reset": False,
        "max_frames": 1e5,
        "learning_rate": 1e-3,
        "act_strategy": "e_greedy",  # e_greedy or softmax
        "epsilon_decay": 2 * 1e4,
        "epsilon_final": 0.1,
        "batch_size": 32,
        "buffer_size": 10000,
        "use_lag_agent": True,
        "lag_update_freq": 100,
        "save_trained_agent": True,
        "agent_saving_gain": 30,
        "agent_load_score": 46,
        "test_n_games": 10,
        "with_graphics": False,
        "force_cpu": False,
    },
    "PongNoFrameskip-v4": {
        "id": "Pong",
        "rewards_mean_length": 100,
        "is_atari": True,
        "fire_reset": True,
        "max_frames": 1e6,
        "learning_rate": 1e-4,
        "act_strategy": "e_greedy",  # e_greedy or softmax
        "epsilon_decay": 2 * 1e5,
        "epsilon_final": 0.1,
        "batch_size": 32,
        "buffer_size": 10000,
        "use_lag_agent": True,
        "lag_update_freq": 10000,
        "save_trained_agent": True,
        "agent_saving_gain": 3,
        "agent_load_score": 19,
        "test_n_games": 10,
        "with_graphics": False,
        "force_cpu": False,
    },
    "SpaceInvaders-v0": {
        "id": "Space",
        "rewards_mean_length": 100,
        "is_atari": True,
        "fire_reset": True,
        "max_frames": 1e6,
        "learning_rate": 1e-4,
        "act_strategy": "e_greedy",  # e_greedy or softmax
        "epsilon_decay": 2 * 1e5,
        "epsilon_final": 0.1,
        "batch_size": 32,
        "buffer_size": 10000,
        "use_lag_agent": True,
        "lag_update_freq": 10000,
        "save_trained_agent": True,
        "agent_saving_gain": 50,
        "agent_load_score": 15,
        "test_n_games": 10,
        "with_graphics": False,
        "force_cpu": False,
    },
    "MsPacman-v0": {
        "id": "Pac",
        "rewards_mean_length": 100,
        "is_atari": True,
        "fire_reset": False,
        "max_frames": 4*1e6,
        "learning_rate": 1e-4,
        "act_strategy": "softmax",  # e_greedy or softmax
        "epsilon_decay": 2 * 1e5,
        "epsilon_final": 0.1,
        "batch_size": 32,
        "buffer_size": 1e5,
        "use_lag_agent": True,
        "lag_update_freq": 10000,
        "save_trained_agent": True,
        "agent_saving_gain": 250,
        "agent_load_score": 3636,
        "test_n_games": 10,
        "with_graphics": False,
        "force_cpu": False,
    }
}

def get_config(env_name):
    return CONFIG[env_name]
