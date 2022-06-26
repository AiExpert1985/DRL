import random
import torch
import argparse
import gym

import my_lib.my_common as common
import my_lib.my_wrappers as wrappers
import my_lib.my_dqn_model as dqn_model
import my_lib.my_agents as agent
import my_lib.my_actions as actions

if __name__ == '__main__':
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", default=False, action="store_true", help="force using cpu")
    args = parser.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    env = gym.make(params.env_name)
    env = wrappers.wrap_dqn(env)
    env.seed(common.SEED)
    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = agent.TargetNet(net)
    selector = actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = actions.EpsilonTracker(selector, params)

