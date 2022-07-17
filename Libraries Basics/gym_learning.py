import gym

env = gym.make('ALE/Breakout-v5', render_mode='human')

# Discrete space
action_space = env.action_space
action_space_size = action_space.n
action_meanings = env.unwrapped.get_action_meanings()
print(action_space)
print('action space size:', action_space_size)
print('meaning of each action:', action_meanings)

# Box space is (low, high, shape)
obs_space = env.observation_space
obs_shape = obs_space.shape
obs_high = obs_space.high
obs_low = obs_space.low

print('observation space:', obs_space)
print('observation shape:', obs_shape)
print('observation highest value', obs_high)
print('observation lowest value', obs_low)

env.reset()
while True:
    action = action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()
