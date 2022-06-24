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


class Agent(nn.Module):
    def __init__(self, device, n_actions, act_strategy, exp_buffer):
        super().__init__()
        self.n_actions = n_actions
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
        avail_actions = list(range(model.n_actions))
        action = np.random.choice(avail_actions, p=probs)
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


def save_agent(prev_frame, agent, optimizer, rewards, config):
    best_rewards_mean = np.mean(rewards)
    print(f"--------------> Saving agent with score {int(best_rewards_mean)}")
    file_path = f"best_models/{config['id']}_{int(best_rewards_mean)}.pth"
    checkpoint = {'frame': prev_frame,
                  'agent_state_dict': agent.state_dict(),
                  'exp_buffer': agent.exp_buffer,
                  'act_strategy': agent.act_strategy,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'rewards': rewards,
                  'best_reward_mean': best_rewards_mean,
                  }
    torch.save(checkpoint, file_path)


def load_agent(agent, optimizer, config):
    file_path = f"best_models/{config['id']}_{config['agent_load_score']}.pth"
    check_point = torch.load(file_path)
    start_frame = check_point['frame']
    agent.load_state_dict(check_point['agent_state_dict'])
    agent.exp_buffer = check_point['exp_buffer']
    agent.act_strategy = check_point['act_strategy']
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    train_rewards = check_point['rewards']
    best_reward_mean = check_point['best_reward_mean']
    return start_frame, agent, optimizer, train_rewards, best_reward_mean


def train(env, agent, optimizer, device, config, agent_mode):
    train_rewards = []
    start_frame = 0
    best_rewards_mean = 1e-100  # assign very small number
    if agent_mode == "resume":
        start_frame, agent, optimizer, train_rewards, best_rewards_mean = load_agent(agent, optimizer, config)
        print("**************** Training Resumed ****************")
    tb_title = f"-MyDQNv1_{config['id']}_lag={config['use_lag_agent']}_stgy={config['act_strategy']: .0f}" \
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
            speed = (frame - prev_frame) / (time.time() - prev_time)
            prev_frame = frame
            prev_time = time.time()
            print(f"{frame}: r = {train_rewards[-1]:.0f}, r_mean = {int(rewards_mean)}, "
                  f"speed = {int(speed)} f/s")
            writer.add_scalar("100_rewards_mean", rewards_mean, frame)
            writer.add_scalar("episode_reward", train_rewards[-1], frame)
            writer.add_scalar("speed", speed, frame)
            if config['act_strategy'] == 'e_greedy':
                writer.add_scalar("epsilon", agent.act_strategy.val, frame)
            if rewards_mean > best_rewards_mean:
                if config['save_trained_agent'] and (rewards_mean - saved_agent_reward) > config['agent_saving_gain']:
                    save_agent(frame, agent, optimizer, train_rewards[-mean_length:], config)
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
    start_frame, agent, optimizer, train_rewards, best_rewards_mean = load_agent(agent, optimizer, config)
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


def set_game(env_id, agent_mode):
    config = CONFIG[env_id]
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
        "batch_size": 64,
        "buffer_size": 1e5,
        "use_lag_agent": True,
        "lag_update_freq": 10000,
        "save_trained_agent": True,
        "agent_saving_gain": 250,
        "agent_load_score": 3417,
        "test_n_games": 10,
        "with_graphics": False,
        "force_cpu": False,
    }
}

if __name__ == "__main__":
    id_ = "MsPacman-v0"
    mode = "train"
    set_game(id_, mode)
