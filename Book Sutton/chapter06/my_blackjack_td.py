import numpy as np
from tqdm import tqdm
from collections import defaultdict

DECK = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
CARD_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
              'J': 10, 'Q': 10, 'K': 10, 'A': 11}

TERMINAL_STATE = (0, False, 0)


HIT = 0
STAND = 1
ACTIONS = [HIT, STAND]

EPSILON = 0.1
GAMMA = 0.9
ALPHA = 0.01

V = defaultdict(lambda: [0, 0])                             # state: [hit_value, stand_value]
Pi = defaultdict(lambda: np.random.choice(ACTIONS))         # state: action
N = defaultdict(int)                                        # (state, action): visit_count


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
    if player_sum == 22:
        player_sum = 12
    dealer_card_val = CARD_VALUE[pick_card()]
    initial_state = (player_sum, usable_ace, dealer_card_val)
    return initial_state


def player_behavior_policy(state):
    return np.random.choice(ACTIONS) if np.random.rand() < EPSILON else Pi[state]


def player_target_policy(state):
    return Pi[state]


def dealer_policy(dealer_sum):
    action = STAND if dealer_sum >= 17 else HIT
    return action


def player_turn(player_policy, state):
    trajectory = []
    player_sum, usable_ace, dealer_card_val = state
    while player_sum <= 21:
        action = player_policy(state)
        if action == STAND:
            trajectory.append([state, action, TERMINAL_STATE])
            break
        player_card = pick_card()
        player_sum, usable_ace = evaluate_hand(player_sum, usable_ace, player_card)
        if player_sum > 21:
            trajectory.append([state, action, TERMINAL_STATE])
            break
        next_state = (player_sum, usable_ace, dealer_card_val)
        trajectory.append([state, action, next_state])
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
    if dealer_sum > player_sum:
        result = -1
    elif player_sum > dealer_sum:
        result = 1
    else:
        result = 0
    return result

def show_game(player_sum, dealer_sum, result, trajectory):
    print("***** New Game *****")
    print(f"    player_sum = {player_sum}")
    print(f"    dealer_sum = {dealer_sum}")
    print(f"    result = {result}")
    print(f"    {trajectory}")

def play_game(player_policy, print_game=False):
    initial_state =  initialize_game()
    trajectory, player_sum = player_turn(player_policy, initial_state)
    dealer_sum = None
    if player_sum > 21:
        result = -1
    else:
        dealer_card_val = initial_state[2]
        dealer_sum = dealer_turn(dealer_card_val)
        result = 1 if dealer_sum > 21 else game_result(player_sum, dealer_sum)
    rewards = [0] * (len(trajectory) - 1) + [result]
    for i, t in enumerate(trajectory):
        t.append(rewards[i])
    if print_game:
        show_game(player_sum, dealer_sum, result, trajectory)
    return trajectory, result


def td(trajectory):
    trajectory.reverse()
    for s, a, s_next, r in trajectory:
        target = r + GAMMA * np.max(V[s_next])
        td_error = target - V[s][a]
        V[s][a] += ALPHA * td_error
        Pi[s] = np.random.choice(np.where(V[s] == np.max(V[s]))[0])
        N[(s, a)] += 1


def train(total_games):
    print("training started ....")
    results = []
    for _ in tqdm(range(total_games)):
        trajectory, result = play_game(player_behavior_policy)
        td(trajectory)
        results.append(result)
    return results


def test(num_games):
    print("testing started ...")
    results = []
    for _ in tqdm(range(num_games)):
        _, result = play_game(player_target_policy, print_game=False)
        results.append(result)
    return np.mean(results)


def run_simulation(train_episodes, test_episodes):
    train(train_episodes)
    result = test(test_episodes)
    print(f"player's mean score of {test_episodes} test games = {np.round(result, 2)}")
    print_order_states()


def print_order_states():
    for key in sorted(V.keys(), reverse=True):
        val = V[key]
        print(f'{key}: [{round(val[0], 2)}, {round(val[1], 2)}]; '
              f'policy: {Pi[key]}; '
              f'counts: [{N[(key, 0)]}, {N[(key, 1)]}]')


if __name__ == '__main__':
    n_train = 1000000
    n_test = 100000
    run_simulation(n_train, n_test)
