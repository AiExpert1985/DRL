"""
chapter 05 of book "Introduction to Reinforcement learning" by Sutton-2018
blackjack game implementation using:
- Monte carlo method
- off policy
- importance sampling (both ordinary and weighted)
- incremental implementation
"""

import numpy as np
from tqdm import tqdm
from collections import defaultdict

DECK = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
CARD_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
              'J': 10, 'Q': 10, 'K': 10, 'A': 11}


HIT = 0
STAND = 1
ACTIONS = [HIT, STAND]

EPSILON = 0.1
GAMMA = 1.0
DISCOUNTS = [GAMMA ** i for i in range(12)]               # max steps in each game can't exceed 11

policy = defaultdict(lambda: np.random.choice(ACTIONS))   # state: action
V = defaultdict(lambda: [0, 0])                 # state: [hit_value, stand_value]
N = defaultdict(int)                     # (state, action): visit_count
C = defaultdict(int)                          # (state,action): sum of imp_ratio at time t


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


def player_behavior_policy(state):
    return np.random.choice(ACTIONS) if np.random.rand() < EPSILON else policy[state]


def player_behavior_policy_prob(state, action):
    return (1-EPSILON) + (EPSILON/len(ACTIONS)) if action == policy[state] else EPSILON/len(ACTIONS)


def player_target_policy(state):
    return policy[state]


def player_target_policy_prob(state, action):
    return 1.0 if action == policy[state] else 0.0


def dealer_policy(dealer_sum):
    action = STAND if dealer_sum >= 17 else HIT
    return action


def player_turn(player_policy, state):
    trajectory = []
    player_sum, usable_ace, dealer_card_val = state
    while True:
        action = player_policy(state)
        trajectory.append((state, action))
        if action == STAND:
            break
        player_card = pick_card()
        player_sum, usable_ace = evaluate_hand(player_sum, usable_ace, player_card)
        if player_sum > 21:
            break
        state = (player_sum, usable_ace, dealer_card_val)
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


def play_game(player_policy, initial_state=None):
    initial_state = initial_state if initial_state is not None else initialize_game()
    trajectory, player_sum = player_turn(player_policy, initial_state)
    if player_sum > 21:
        result = -1
    else:
        dealer_card_val = initial_state[2]
        dealer_sum = dealer_turn(dealer_card_val)
        result = 1 if dealer_sum > 21 else game_result(player_sum, dealer_sum)
    return trajectory, result


def monte_carlo_policy_iteration(trajectory, reward):
    version = 'weighted'
    trajectory.reverse()
    W = 1
    for (state, action), discount in zip(trajectory, DISCOUNTS[:len(trajectory)]):
        N[(state, action)] += 1
        G = discount * reward
        if version == 'ordinary':
            V[state][action] += 1/N[(state, action)] * (G * W - V[state][action])
        if version == 'weighted':
            if W == 0:
                break
            C[(state, action)] += W
            V[state][action] += W / C[(state, action)] * (G - V[state][action])
        policy[state] = np.argmax(V[state])
        if policy[state] != action:
            break
        W *= player_target_policy_prob(state, action) / player_behavior_policy_prob(state, action)


def train(total_games):
    print("training started ...")
    results = []
    for _ in tqdm(range(total_games)):
        trajectory, result = play_game(player_behavior_policy)
        monte_carlo_policy_iteration(trajectory, result)
        results.append(result)
    return results


def test(num_games):
    print("testing started ....")
    results = []
    for _ in tqdm(range(num_games)):
        _, result = play_game(player_target_policy)
        results.append(result)
    return np.mean(results)


def run_simulation(total_games):
    train(total_games)
    result = test(100000)
    print("player's mean score of last 100 games = ", np.round(result, 2))
    print_order_states()


def print_order_states():
    for state in sorted(V.keys(), reverse=True):
        val = V[state]
        print(f'{state}: [{round(val[0], 2)}, {round(val[1], 2)}]; '
              f'policy: {player_target_policy(state)}; '
              f'counts: [{N[(state, 0)]}, {N[(state, 1)]}]')


if __name__ == '__main__':
    n_games = 100000
    run_simulation(n_games)
