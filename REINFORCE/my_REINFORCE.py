import torch
from torch import nn, optim, distributions
import gym
import numpy as np
from tensorboardX import SummaryWriter
import time

MAX_EPISODES = 400
GAMMA = 0.99
HIDDEN_DIM = 64
MAX_STEPS = 400


class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, out_dim)
        ]
        self.model = nn.Sequential(*layers)
        self.log_probs = []
        self.rewards = []
        self.train()

    def reset_records(self):
        self.log_probs = []
        self.rewards = []

    def update_rewards(self, reward):
        self.rewards.append(reward)

    def forward(self, x):
        return self.model(x)

    def act(self, state):
        state = torch.from_numpy(state).float()
        logits = self(state)
        pd = distributions.Categorical(logits=logits)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()


def run_episode(env, agent):
    time.sleep(1)  # pause rendering for 1 sec after each episode
    state = env.reset()
    agent.reset_records()
    for _ in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        agent.update_rewards(reward)
        if done:
            break
        state = next_state


def loss_fn(agent):
    rets = []
    future_rets = 0.0
    for rew in agent.rewards:
        future_rets = rew + (GAMMA * future_rets)
        rets.append(future_rets)
    rets.reverse()
    rets = torch.tensor(rets, dtype=torch.float32)
    log_probs = torch.stack(agent.log_probs)
    loss = - torch.sum(log_probs * rets)
    return loss


def run():
    writer = SummaryWriter(comment="-My_Foundation_REINFORCE_v1")
    episode_rewards = []
    env = gym.make("CartPole-v1")
    agent = Pi(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(params=agent.parameters(), lr=0.01)
    for episode in range(MAX_EPISODES):
        run_episode(env, agent)
        loss = loss_fn(agent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episode_rewards.append(len(agent.rewards))
        reward_avg = np.mean(episode_rewards[-10:])
        writer.add_scalar("loss", loss.item(), episode)
        writer.add_scalar("last_10_rewards", reward_avg, episode)
        print(f"{episode}, Loss = {loss.item(): .2f}, Reward = {len(agent.rewards)}, Last_10_AVG = {reward_avg: .0f}")
        # if reward_avg > 199:
        #     print(f"Solved in {episode} episodes !")
        #     break
    writer.flush()
    writer.close()


if __name__ == "__main__":
    run()
