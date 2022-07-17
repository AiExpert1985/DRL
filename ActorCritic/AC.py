# actor-critic
# start: 2022-7-16
# last update: 2022-7-16
import torch.cuda

import wrappers
import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()

        self.n_actions = n_actions
        self.avail_actions = list(range(n_actions))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    env_id = "ALE/Breakout-v5"
    env = wrappers.make_env(env_id)
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape

    ac = ActorCritic(state_space_size, action_space_size)

    obs = env.reset()
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    print(ac(obs).shape)

