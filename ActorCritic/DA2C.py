# actor-critic
# start: 2022-7-16
# last update: 2022-7-17

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
            nn.Linear(cnn_out_len, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(cnn_out_len, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
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


def get_action(model, obs, device):
    obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    with torch.no_grad():
        actor = model(obs)[0]
        probs = actor.squeeze().cpu().numpy()
    action = np.random.choice(model.avail_actions, p=probs)
    return action


def get_batch(history, device):
    observs, actions, rewards, next_observs, dones = zip(*history)
    observs = torch.from_numpy(np.stack(observs)).float().to(device)
    actions = torch.tensor(actions).unsqueeze(1).long().to(device)
    rewards = torch.tensor(rewards).unsqueeze(1).float().to(device)
    next_observs = torch.from_numpy(np.stack(next_observs)).float().to(device)
    dones = torch.tensor(dones).unsqueeze(1).bool().to(device)
    return observs, actions, rewards, next_observs, dones


def train(batch, model, optimizer, gamma=0.98):
    model.train()
    observs, actions, rewards, next_obervs, dones = batch
    actor, V_ref = model(observs)
    action_probs = torch.gather(actor, dim=1, index=actions).squeeze()
    with torch.no_grad():
        V_prime = model(next_obervs)[1]
        V_prime[dones] = 0
    V = rewards + gamma * V_prime
    actor_loss = - torch.log(action_probs) * (V - V_ref.detach())
    critic_loss = torch.pow(V - V_ref, 2)
    loss = torch.mean(critic_loss + actor_loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return actor_loss + critic_loss


def run():
    env_id = "ALE/Breakout-v5"
    env = wrappers.make_env(env_id)
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ActorCritic(state_space_size, action_space_size).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    writer = SummaryWriter(comment="-A2C")

    max_frames = 1000000
    batch_size = 256

    history = []
    rewards = []
    episode_reward = 0.0
    obs = env.reset()
    for frame in range(max_frames):
        action = get_action(model, obs, device)
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        history.append((obs, action, reward, next_obs, done))
        obs = next_obs
        if done:
            rewards.append(episode_reward)
            episode_reward = 0.0
            obs = env.reset()
        if len(history) < batch_size:
            continue
        reward_100 = np.mean(rewards[-100:])
        batch = get_batch(history, device)
        loss = train(batch, model, optimizer)
        writer.add_scalar("reward_100", reward_100, frame)
        print(f'{frame}: reward = {reward_100: .1f}, loss = {loss.item(): .3f}')
        history = []
    writer.flush()
    writer.close()


if __name__ == "__main__":
    run()
