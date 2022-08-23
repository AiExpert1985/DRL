import numpy as np
from tqdm import tqdm
from collections import defaultdict

DECK = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
CARD_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
              'J': 10, 'Q': 10, 'K': 10, 'A': 11}


HIT = 0
STAND = 1
ACTIONS = [HIT, STAND]

GAMMA = 0.9
DISCOUNTS = [GAMMA ** i for i in range(12)]                 # max steps in each game can't exceed 11

state_value = defaultdict(lambda: [0, 0])                   # state: [hit_value, stand_value]
policy = defaultdict(lambda: np.random.choice(ACTIONS))     # state: action
state_action_count = defaultdict(int)                 # (state, action): visit_count


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


def player_policy(state):
    return policy[state]


def dealer_policy(dealer_sum):
    action = STAND if dealer_sum >= 17 else HIT
    return action


def player_turn(state, initial_action=None):
    trajectory = []
    player_sum, usable_ace, dealer_card_val = state
    action = initial_action if initial_action is not None else player_policy(state)
    while True:
        trajectory.append((state, action))
        if action == STAND:
            break
        player_card = pick_card()
        player_sum, usable_ace = evaluate_hand(player_sum, usable_ace, player_card)
        if player_sum > 21:
            break
        state = (player_sum, usable_ace, dealer_card_val)
        action = player_policy(state)
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


def play_game(initial_state=None, player_initial_action=None):
    initial_state = initial_state if initial_state is not None else initialize_game()
    trajectory, player_sum = player_turn(initial_state, player_initial_action)
    if player_sum > 21:
        result = -1
    else:
        dealer_card_val = initial_state[2]
        dealer_sum = dealer_turn(dealer_card_val)
        result = 1 if dealer_sum > 21 else game_result(player_sum, dealer_sum)
    return trajectory, result


def monte_carlo_policy_iteration(trajectory, reward):
    trajectory.reverse()
    for (s, a), discount in zip(trajectory, DISCOUNTS[:len(trajectory)]):
        G = discount * reward
        n = state_action_count[(s, a)] + 1
        Qs = state_value[s]
        Qs[a] += 1/n * (G - state_value[s][a])
        state_action_count[(s, a)] = n
        policy[s] = np.random.choice(np.where(Qs == np.max(Qs))[0])


def train(total_games):
    results = []
    for _ in tqdm(range(total_games)):
        trajectory, result = play_game(player_initial_action=np.random.choice(ACTIONS))
        monte_carlo_policy_iteration(trajectory, result)
        results.append(result)
    return results


def test(num_games):
    results = []
    for _ in range(num_games):
        _, result = play_game()
        results.append(result)
    return np.mean(results)


def run_simulation(total_games):
    train(total_games)
    result = test(1000000)
    print("player's mean score of last 100 games = ", np.round(result, 2))
    print_order_states()


def print_order_states():
    for key in sorted(state_value.keys(), reverse=True):
        val = state_value[key]
        print(f'{key}: [{round(val[0], 2)}, {round(val[1], 2)}]; policy: {policy[key]}; '
              f'counts: [{state_action_count[(key, 0)]}, {state_action_count[(key, 1)]}]')


if __name__ == '__main__':
    n_games = 1000000
    run_simulation(n_games)
