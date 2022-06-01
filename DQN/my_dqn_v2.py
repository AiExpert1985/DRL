import gym
import wrappers
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
import numpy as np
from collections import deque
import random
import copy


class Agent(nn.Module):
    def __init__(self, n_actions, epsilon, exp_buffer):
        super().__init__()
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.exp_buffer = exp_buffer

    @torch.no_grad()
    def act(self, state, device):
        if random.random() < self.epsilon.val:
            action = random.choice(range(self.n_actions))
        else:
            state = torch.from_numpy(state).unsqueeze(0).float().to(device)
            logits = self.forward(state)
            action = (torch.max(logits, dim=1)[1]).item()
        return action

    def act_randomly(self):
        return random.choice(range(self.n_actions))


class FcAgent(Agent):
    def __init__(self, input_shape, n_actions, epsilon, exp_buffer, hidden_dim=128):
        super().__init__(n_actions, epsilon, exp_buffer)
        fc_layers = [
            nn.Linear(input_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions)
        ]
        self.full_connected = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.full_connected(x)


class CnnAgent(Agent):
    # noinspection PyTypeChecker
    def __init__(self, input_shape, n_actions, epsilon, exp_buffer, hidden_dim=512):
        super().__init__(n_actions, epsilon, exp_buffer)
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

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class Epsilon:
    def __init__(self, start=1.0, final=0.01, decay=500):
        self.start = start
        self.final = final
        self.decay = decay
        self.n_frames = 0
        self.val = self.start

    def update(self):
        self.n_frames += 1
        self.val = max(self.final, (self.start - (self.n_frames / self.decay)))


class ExperienceBuffer:
    def __init__(self, capacity=1000, sample_len=32):
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
    agent_score = int(best_rewards_mean / config['agent_saving_score_normalizer'])
    file_path = f"best_models/{config['id']}_{agent_score}.pth"
    checkpoint = {'frame': prev_frame,
                  'agent_state_dict': agent.state_dict(),
                  'exp_buffer': agent.exp_buffer,
                  'epsilon': agent.epsilon,
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
    agent.epsilon = check_point['epsilon']
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
    tb_title = f"-MyDQNv1_{config['id']}_lag={config['use_lag_agent']}_eps={config['epsilon_decay']: .0f}" \
               f"_lr={config['learning_rate']}_batch={config['batch_size']}"
    writer = SummaryWriter(comment=tb_title)
    lag_agent = copy.deepcopy(agent) if config["use_lag_agent"] else agent
    state = env.reset()
    episode_reward = 0
    saved_agent_reward = best_rewards_mean
    mean_length = config['rewards_mean_length']
    for frame in range(start_frame, int(config["max_frames"])):
        if config['random_base_line']:
            action = agent.act_randomly()
        else:
            action = agent.act(state, device)
        next_state, reward, done, _ = env.step(action)
        if not config['is_atari'] and config['with_graphics']:
            env.render()
        episode_reward += reward
        experience = (torch.from_numpy(state), action, reward, torch.from_numpy(next_state), done)
        agent.exp_buffer.push(experience)
        if done:
            state = env.reset()
            train_rewards.append(episode_reward)
            episode_reward = 0
            rewards_mean = np.mean(train_rewards[-mean_length:])
            print(f"{frame}: r = {train_rewards[-1]:.0f}, r_mean = {rewards_mean:.1f}, eps = {agent.epsilon.val:.2f}")
            writer.add_scalar("100_rewards_mean", rewards_mean, frame)
            writer.add_scalar("epsilon", agent.epsilon.val, frame)
            writer.add_scalar("episode_reward", train_rewards[-1], frame)
            if rewards_mean > best_rewards_mean:
                if config['save_trained_agent'] and (rewards_mean - saved_agent_reward) > config['agent_saving_gain']:
                    save_agent(frame, agent, optimizer, train_rewards[-mean_length:], config)
                best_rewards_mean = rewards_mean
        else:
            state = next_state
        if not agent.exp_buffer.ready():
            continue
        optimizer.zero_grad()
        loss = calculate_loss(agent, lag_agent, device)
        loss.backward()
        optimizer.step()
        agent.epsilon.update()
        if config['use_lag_agent'] and frame % 1000 == 0:
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


def set_game(env_id, agent_mode):
    config = CONFIG[env_id]
    env = wrappers.make_env(env_id, config['with_graphics']) if config['is_atari'] else gym.make(env_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dims = env.observation_space.shape
    output_dim = env.action_space.n
    epsilon = Epsilon(start=1.0, final=0.01, decay=config["epsilon_decay"])
    ex_buffer = ExperienceBuffer(capacity=10000, sample_len=config["batch_size"])
    AgentClass = CnnAgent if config['is_atari'] else FcAgent
    agent = AgentClass(input_dims, output_dim, epsilon, ex_buffer).to(device)
    optimizer = optim.Adam(params=agent.parameters(), lr=config['learning_rate'])
    if agent_mode == "train" or agent_mode == "resume":
        train(env, agent, optimizer, device, config, agent_mode)
    if agent_mode == "test":
        test(env, agent, optimizer, device, config)


CONFIG = {
    "CartPole-v1": {
        "random_base_line": True,
        "rewards_mean_length": 100,
        "is_atari": False,
        "max_frames": 1e5,
        "learning_rate": 0.001,
        "epsilon_decay": 2 * 1e4,
        "batch_size": 32,
        "use_lag_agent": True,
        "save_trained_agent": True,
        "agent_saving_score_normalizer": 10,
        "agent_saving_gain": 30,
        "id": "Cart",
        "agent_load_score": 46,
        "test_n_games": 10,
        "with_graphics": True,
    },
    "PongNoFrameskip-v4": {
        "random_base_line": True,
        "rewards_mean_length": 100,
        "is_atari": True,
        "max_frames": 1e6,
        "learning_rate": 0.0001,
        "epsilon_decay": 2 * 1e5,
        "batch_size": 32,
        "use_lag_agent": True,
        "save_trained_agent": True,
        "agent_saving_score_normalizer": 1,
        "agent_saving_gain": 3,
        "id": "Pong",
        "agent_load_score": 19,
        "test_n_games": 10,
        "with_graphics": True,
    },
    "SpaceInvaders-v0": {
        "random_base_line": True,
        "rewards_mean_length": 500,
        "is_atari": True,
        "max_frames": 1e6,
        "learning_rate": 0.0001,
        "epsilon_decay": 2 * 1e5,
        "batch_size": 32,
        "use_lag_agent": True,
        "save_trained_agent": True,
        "agent_saving_score_normalizer": 10,
        "agent_saving_gain": 50,
        "id": "Space",
        "agent_load_score": 15,
        "test_n_games": 10,
        "with_graphics": True,
    }
}


if __name__ == "__main__":
    id_ = "SpaceInvaders-v0"
    mode = "train"
    set_game(id_, mode)
