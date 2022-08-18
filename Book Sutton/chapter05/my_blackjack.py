import numpy as np
from tqdm import tqdm

DECK = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
CARD_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
              'J': 10, 'Q': 10, 'K': 10, 'A': 11}
HIT = 0
STAND = 1
ACTIONS = [HIT, STAND]

EPSILON = 0.1
GAMMA = 0.9

state_action_value = {}  # mapping (player_sum, usable_ace, dealer_card_val): [hit_val, stand_val]


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


def player_MC_policy(state):
    if not state_action_value.get(state) or np.random.rand() < EPSILON:
        action = np.random.choice(ACTIONS)
    else:
        action = np.argmax(state_action_value[state])
    return action


def dealer_policy(dealer_sum):
    action = STAND if dealer_sum >= 17 else HIT
    return action


def player_turn(state):
    trajectory = []
    player_sum, usable_ace, dealer_card_val = state
    while True:
        # action = player_fixed_policy(state)
        action = player_MC_policy(state)
        trajectory.append((state, action))
        if action == STAND:
            break
        player_card = pick_card()
        player_sum, usable_ace = evaluate_hand(player_sum, usable_ace, player_card)
        state = (player_sum, usable_ace, dealer_card_val)
    return trajectory, player_sum


def dealer_turn(card_val):
    usable_ace = False
    if card_val == 11:
        usable_ace = True
    dealer_sum = card_val
    while True:
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


def play_game():
    initial_state = initialize_game()
    trajectory, player_sum = player_turn(initial_state)
    if player_sum > 21:
        return trajectory, -1
    dealer_card_val = initial_state[2]
    dealer_sum = dealer_turn(dealer_card_val)
    if dealer_sum > 21:
        return trajectory, 1
    return trajectory, game_result(player_sum, dealer_sum)


def monte_carlo_update(trajectory, result):
    Q_next = 0
    trajectory.reverse()
    for state, action in trajectory:
        if state_action_value.get(state):
            Qs = state_action_value[state]
            Qs[action] += result + GAMMA * Q_next
            Q_next = Qs[action]
        else:
            state_action_value[state] = [0, 0]


def run_simulation(n_games):
    results = []
    for _ in tqdm(range(n_games)):
        trajectory, result = play_game()
        monte_carlo_update(trajectory, result)
        results.append(result)
    print("player's mean score = ", np.round(np.mean(results), 3))


if __name__ == '__main__':
    num_games = 100000
    run_simulation(num_games)
