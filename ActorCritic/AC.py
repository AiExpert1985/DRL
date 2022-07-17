# actor-critic
# start: 2022-7-16
# last update: 2022-7-16

import wrappers
import torch
from torch import nn

import numpy as np
from tensorboardX import SummaryWriter


class ActorCritic(nn.Module):
    # noinspection PyTypeChecker
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
            nn.Linear(512, self.n_actions),
            nn.Softmax(dim=1)
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
        flat_cnn_out = cnn_out.view(cnn_out.size()[0], -1)
        actor = self.actor(flat_cnn_out)
        critic = self.critic(flat_cnn_out.detach())
        return actor, critic


def get_action(model, obs):
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    with torch.no_grad():
        actor, critic = model(obs)
        probs = actor.squeeze().cpu().numpy()
    action = np.random.choice(model.avail_actions, p=probs)
    return action


def run_episode(env, model):
    obs = env.reset()
    traj = []
    while True:
        action = get_action(model, obs)
        next_obs, reward, done, _ = env.step(action)
        traj.append((obs, action, reward, next_obs, done))
        if done:
            return traj
        obs = next_obs


def get_batch(history, device):
    observs, actions, rewards, next_observs, dones = zip(*history)
    observs = torch.from_numpy(np.stack(observs)).float().to(device)
    actions = torch.tensor(actions).unsqueeze(1).long()
    rewards = torch.tensor(rewards).unsqueeze(1).float()
    next_observs = torch.from_numpy(np.stack(next_observs)).float().to(device)
    dones = torch.tensor(dones).unsqueeze(1).bool()
    return observs, actions, rewards, next_observs, dones


def calculate_loss(batch, model):
    observs, actions, rewards, next_obervs, dones = batch
    actor, V = model(observs)
    action_probs = torch.gather(actor, dim=1, index=actions).squeeze()
    with torch.no_grad():
        V_prime = model(next_obervs)[1]
        V_prime[dones] = 0
    actor_loss = - torch.mean(torch.log(action_probs) * (rewards + V_prime - V.detach()))
    critic_loss = torch.mean(torch.square(rewards + V_prime - V))
    return actor_loss + critic_loss


def run():
    env_id = "ALE/Breakout-v5"
    env = wrappers.make_env(env_id)
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ActorCritic(state_space_size, action_space_size).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    writer = SummaryWriter(comment="actor_critic")

    n_epochs = 100
    n_episodes_per_epoch = 10

    for i in range(n_epochs):
        history = []
        for j in range(n_episodes_per_epoch):
            traj = run_episode(env, model)
            history.extend(traj)

        batch = get_batch(history, device)
        reward_mean = torch.mean(batch[2])
        loss = calculate_loss(batch, model)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar("reward_mean", reward_mean, i)

        print(i)


if __name__ == "__main__":
    run()
