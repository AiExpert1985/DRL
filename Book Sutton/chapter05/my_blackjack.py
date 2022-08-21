import numpy as np
from tqdm import tqdm
from collections import defaultdict

DECK = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
CARD_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
              'J': 10, 'Q': 10, 'K': 10, 'A': 11}


HIT = 0
STAND = 1
ACTIONS = [HIT, STAND]

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
GAMMA = 0.9
ALPHA = 0.1

TERMINAL_STATE = (0, False, 0)
state_action_value = defaultdict(lambda: [0, 0])


def pick_card():
    return np.random.choice(DECK)


def evaluate_card(card):
    return CARD_VALUE[card]


def evaluate_hand(cards_sum, usable_ace, card):
    cards_sum += evaluate_card(card)
    if cards_sum > 21 and (usable_ace or card == 'A'):
        cards_sum -= 10
        usable_ace = False
    if usable_ace and card == 'A':
        usable_ace = True
    return cards_sum, usable_ace


def initialize_game():
    usable_ace = False
    player_cards = [pick_card() for _ in range(2)]
    if 'A' in player_cards:
        usable_ace = True
    player_sum = np.sum([evaluate_card(card) for card in player_cards])
    # if player has 2 aces, use one of them (i.e. consider one of them as 1 instead of 11)
    if player_sum == 22:
        player_sum = 12
    dealer_card_val = CARD_VALUE[pick_card()]
    initial_state = (player_sum, usable_ace, dealer_card_val)
    return initial_state


def player_fixed_policy(state):
    player_sum, usable_ace, dealer_card_val = state
    action = STAND if player_sum >= 15 else HIT
    return action


def player_MC_policy(state, epsilon):
    if not state_action_value.get(state) or np.random.rand() < epsilon:
        action = np.random.choice(ACTIONS)
        print("Random", action)
    else:
        action = np.argmax(state_action_value[state])
        print("Best", action)
    return action


def dealer_policy(dealer_sum):
    action = STAND if dealer_sum >= 17 else HIT
    return action


def player_turn(state, epsilon):
    print("**************************** player turn started", "with epsilon =", epsilon)
    trajectory = []
    player_sum, usable_ace, dealer_card_val = state
    while True:
        print("state", state)
        print("state value", state_action_value[state])
        # action = player_fixed_policy(state)
        action = player_MC_policy(state, epsilon)
        if action == STAND:
            trajectory.append((state, action, TERMINAL_STATE))
            break
        player_card = pick_card()
        print("card", player_card)
        player_sum, usable_ace = evaluate_hand(player_sum, usable_ace, player_card)
        if player_sum > 21:
            trajectory.append((state, action, TERMINAL_STATE))
            break
        next_state = (player_sum, usable_ace, dealer_card_val)
        print("next_state", next_state)
        trajectory.append((state, action, next_state))
        state = next_state
    return trajectory, player_sum


def dealer_turn(card_val):
    usable_ace = False
    if card_val == 11:
        usable_ace = True
    dealer_sum = card_val
    while dealer_sum <= 22:
        card_val = pick_card()
        dealer_sum, usable_ace = evaluate_hand(dealer_sum, usable_ace, card_val)
        action = dealer_policy(dealer_sum)
        if action == STAND:
            break
    return dealer_sum


def game_result(player_sum, dealer_sum):
    if player_sum > dealer_sum:
        result = 1
    elif player_sum < dealer_sum:
        result = -1
    else:
        result = 0
    return result


def play_game(epsilon):
    initial_state = initialize_game()
    trajectory, player_sum = player_turn(initial_state, epsilon)
    if player_sum > 21:
        result = -1
    else:
        dealer_card_val = initial_state[2]
        dealer_sum = dealer_turn(dealer_card_val)
        result = 1 if dealer_sum > 21 else game_result(player_sum, dealer_sum)
    return trajectory, result


def monte_carlo_update(trajectory, rewards):
    # print("**************************** a game started **************************")
    # print('game result =', result)
    trajectory.reverse()
    rewards.reverse()
    for (state, action, next_state), reward in zip(trajectory, rewards):
        # print(state, action, next_state, reward)
        state_action_vals = state_action_value[state]
        # print('state, action = ', state, action)
        # print('state_action_vals', state_action_vals)
        # print('V_next', V_next)
        V_next = np.max(state_action_value[next_state])
        state_action_vals[action] += ALPHA * (reward - GAMMA * V_next)
        # print('state_action_val', state_action_vals[action])
        state_action_value[state] = state_action_vals
        # print("======================================")


def run_simulation(n_games):
    results = []
    epsilon_decay = n_games / 4
    for i in tqdm(range(n_games)):
        epsilon = max(EPSILON_FINAL, (EPSILON_START - (i / epsilon_decay)))
        trajectory, result = play_game(epsilon)
        print(f"====================== game result = {result} =======================")
        rewards = [0] * len(trajectory)
        rewards[-1] = result
        monte_carlo_update(trajectory, rewards)
        results.append(result)
    print("player's mean score of last 100 games = ", np.round(np.mean(results[-1000:]), 3))
    print(len(state_action_value))
    # for key, val in state_action_value.items():
    #     print(key, ":", val)


if __name__ == '__main__':
    num_games = 10000
    run_simulation(num_games)
