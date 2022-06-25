import gym
import wrappers
import torch
from torch import nn, optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from collections import deque
import random
import copy
import time
from my_config import get_config


class Agent(nn.Module):
    def __init__(self, device, n_actions, act_strategy, exp_buffer):
        super().__init__()
        self.n_actions = n_actions
        self.avail_actions = list(range(self.n_actions))
        self.act_strategy = act_strategy
        self.exp_buffer = exp_buffer
        self.device = device

    def act(self, state):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        return self.act_strategy.act(self, state)


class FcAgent(Agent):
    def __init__(self, device, input_shape, n_actions, epsilon, exp_buffer, hidden_dim=128):
        super().__init__(device, n_actions, epsilon, exp_buffer)
        fc_layers = [
            nn.Linear(input_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions)
        ]
        self.full_connected = nn.Sequential(*fc_layers)
        self.to(self.device)

    def forward(self, x):
        return self.full_connected(x)


class CnnAgent(Agent):
    def __init__(self, device, input_shape, n_actions, epsilon, exp_buffer, hidden_dim=512):
        super().__init__(device, n_actions, epsilon, exp_buffer)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        self.to(self.device)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ActionStrategy:
    def act(self, model, state):
        raise NotImplementedError

class EGreedyStrategy(ActionStrategy):
    def __init__(self, start=1.0, final=0.01, decay=500):
        self.start = start
        self.final = final
        self.decay = decay
        self.n_frames = 0
        self.val = self.start

    def update(self):
        self.n_frames += 1
        self.val = max(self.final, (self.start - (self.n_frames / self.decay)))

    @torch.no_grad()
    def act(self, model, state):
        if random.random() < self.val:
            action = random.choice(range(model.n_actions))
        else:
            logits = model.forward(state)
            action = (torch.max(logits, dim=1)[1]).item()
        self.update()
        return action


class SoftMaxStrategy(ActionStrategy):
    @torch.no_grad()
    def act(self, model, state):
        logits = model.forward(state)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        action = np.random.choice(model.avail_actions, p=probs)
        return action


class ExperienceBuffer:
    def __init__(self, capacity=10000, sample_len=32):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.sample_len = sample_len

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self):
        indexes = np.random.choice(range(len(self)), self.sample_len)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indexes])
        return states, actions, rewards, next_states, dones

    def ready(self):
        return len(self) >= self.capacity / 2

    def __len__(self):
        return len(self.buffer)


def calculate_loss(agent, lag_agent, device, gamma=0.99):
    states, actions, rewards, next_states, dones = agent.exp_buffer.sample()
    states = torch.stack(states).float().to(device)
    actions = torch.tensor(actions).unsqueeze(1).long().to(device)
    rewards = torch.tensor(rewards).float().to(device)
    next_states = torch.stack(next_states).float().to(device)
    dones = torch.tensor(dones).bool().to(device)
    with torch.no_grad():
        Qs = lag_agent(next_states)
        Q = torch.max(Qs, dim=1)[0]
        Q[dones] = 0
        V = rewards + gamma * Q
    V_pred = torch.gather(agent(states), index=actions, dim=1).squeeze()
    loss = nn.MSELoss()(V_pred, V)
    return loss


def save_agent(train_duration, prev_frame, agent, optimizer, rewards, config):
    best_rewards_mean = np.mean(rewards)
    print(f"--------------> Saving agent with score {int(best_rewards_mean)}")
    file_path = f"best_models/{config['id']}_{int(best_rewards_mean)}.pth"
    checkpoint = {'train_duration': train_duration,
                  'frame': prev_frame,
                  'agent_state_dict': agent.state_dict(),
                  'act_strategy': agent.act_strategy,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'rewards': rewards,
                  'best_reward_mean': best_rewards_mean,
                  }
    torch.save(checkpoint, file_path)


def load_agent(agent, optimizer, config):
    file_path = f"best_models/{config['id']}_{config['agent_load_score']}.pth"
    check_point = torch.load(file_path)
    train_duration = check_point['train_duration']
    start_frame = check_point['frame']
    agent.load_state_dict(check_point['agent_state_dict'])
    agent.act_strategy = check_point['act_strategy']
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    train_rewards = check_point['rewards']
    best_reward_mean = check_point['best_reward_mean']
    return train_duration, start_frame, agent, optimizer, train_rewards, best_reward_mean


def train(env, agent, optimizer, device, config, agent_mode):
    train_duration = 0
    train_rewards = []
    start_frame = 0
    best_rewards_mean = float('-inf')
    if agent_mode == "resume":
        train_duration, start_frame, agent, optimizer, train_rewards, best_rewards_mean = \
            load_agent(agent, optimizer, config)
        print("**************** Training Resumed ****************")
        print(f"Total training time for the agent = {int(train_duration / 60)} minutes")
    tb_title = f"-MyDQNv3_{config['id']}_lag={config['use_lag_agent']}_stgy={config['act_strategy']}" \
               f"_lr={config['learning_rate']}_batch={config['batch_size']}"
    writer = SummaryWriter(comment=tb_title)
    lag_agent = copy.deepcopy(agent) if config["use_lag_agent"] else agent
    state = env.reset()
    return_ = 0
    saved_agent_reward = best_rewards_mean
    mean_length = config['rewards_mean_length']
    prev_frame = start_frame
    prev_time = time.time()
    for frame in range(start_frame, int(config["max_frames"])):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if not config['is_atari'] and config['with_graphics']:
            env.render()
        return_ += reward
        experience = (torch.from_numpy(state), action, reward, torch.from_numpy(next_state), done)
        agent.exp_buffer.push(experience)
        if done:
            state = env.reset()
            train_rewards.append(return_)
            return_ = 0
            rewards_mean = np.mean(train_rewards[-mean_length:])
            current_time = time.time()
            episode_duration = current_time - prev_time
            speed = (frame - prev_frame) / episode_duration
            prev_frame = frame
            prev_time = current_time
            train_duration += episode_duration
            print(f"{frame}: r = {train_rewards[-1]:.0f}, r_mean = {int(rewards_mean)}, "
                  f"speed = {int(speed)} f/s, train_time = {int(train_duration / 60)} min")
            writer.add_scalar("100_rewards_mean", rewards_mean, frame)
            writer.add_scalar("episode_reward", train_rewards[-1], frame)
            writer.add_scalar("speed", speed, frame)
            if rewards_mean > best_rewards_mean:
                if config['save_trained_agent'] and (rewards_mean - saved_agent_reward) > config['agent_saving_gain']:
                    save_agent(train_duration, frame, agent, optimizer, train_rewards[-mean_length:], config)
                    saved_agent_reward = rewards_mean
                best_rewards_mean = rewards_mean
        else:
            state = next_state
        if not agent.exp_buffer.ready():
            continue
        optimizer.zero_grad()
        loss = calculate_loss(agent, lag_agent, device)
        loss.backward()
        optimizer.step()
        if config['use_lag_agent'] and frame % config['lag_update_freq'] == 0:
            lag_agent.load_state_dict(agent.state_dict())
        writer.add_scalar("loss", loss.item(), frame)
    writer.flush()
    writer.close()


def test(env, agent, optimizer, device, config):
    train_duration, start_frame, agent, optimizer, train_rewards, best_rewards_mean = \
        load_agent(agent, optimizer, config)
    print(f"Agent was trained for {int(train_duration/60)} minutes, with score {config['agent_load_score']}")
    test_rewards = []
    for i in range(config['test_n_games']):
        state = env.reset()
        episode_rewards = 0
        while True:
            state = torch.from_numpy(state).unsqueeze(0).float().to(device)
            Qs = agent(state)
            action = (torch.max(Qs, dim=1)[1]).item()
            next_state, reward, done, _ = env.step(action)
            if not config['is_atari'] and config['with_graphics']:
                env.render()
            state = next_state
            episode_rewards += reward
            if done:
                break
        test_rewards.append(episode_rewards)
        print(f"{i}: episode_reward = {episode_rewards: .0f}, mean_reward = {np.mean(test_rewards): .0f}")

def select_act_strategy(config):
    if config['act_strategy'] == 'e_greedy':
        act_strategy = EGreedyStrategy(start=1.0, final=config['epsilon_final'], decay=config["epsilon_decay"])
    elif config['act_strategy'] == 'softmax':
        act_strategy = SoftMaxStrategy()
    else:
        print("action strategy is not correctly selected")
        raise ValueError
    return act_strategy


def set_game(env_id, agent_mode, config):
    env = wrappers.make_env(env_id, config) if config['is_atari'] else gym.make(env_id)
    if config['force_cpu']:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dims = env.observation_space.shape
    output_dim = env.action_space.n
    ex_buffer = ExperienceBuffer(capacity=int(config['buffer_size']), sample_len=config["batch_size"])
    AgentClass = CnnAgent if config['is_atari'] else FcAgent
    act_strategy = select_act_strategy(config)
    agent = AgentClass(device, input_dims, output_dim, act_strategy, ex_buffer)
    optimizer = optim.Adam(params=agent.parameters(), lr=config['learning_rate'])
    if agent_mode == "train" or agent_mode == "resume":
        train(env, agent, optimizer, device, config, agent_mode)
    if agent_mode == "test":
        test(env, agent, optimizer, device, config)


if __name__ == "__main__":
    id_ = "MsPacman-v0"                # 'CartPole-v1', 'PongNoFrameskip-v4', 'SpaceInvaders-v0', 'MsPacman-v0'
    mode = "train"                     # 'train', 'test', 'resume'
    game_config = get_config(id_)
    set_game(id_, mode, game_config)
