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

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        cnn_out_len = self.get_cnn_out_size(input_shape)

        self.actor = nn.Sequential(
            nn.Linear(cnn_out_len, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(cnn_out_len, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def get_cnn_out_size(self, shape):
        dummy_input = self.conv(torch.zeros(1, *shape))
        return torch.numel(dummy_input)

    def forward(self, x):
        cnn_out = self.conv(x)
        flat_cnn_out = torch.flatten(cnn_out)
        actor = self.actor(flat_cnn_out)
        critic = self.critic(flat_cnn_out.detach())
        return actor, critic


if __name__ == "__main__":
    env_id = "ALE/Breakout-v5"
    env = wrappers.make_env(env_id)
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ac = ActorCritic(state_space_size, action_space_size).to(device)

    obs = env.reset()
    obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    print(ac(obs))

