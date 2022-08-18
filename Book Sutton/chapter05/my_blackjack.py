import numpy as np

DECK = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
CARD_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
              'J': 10, 'Q': 10, 'K': 10, 'A': 11}
HIT = 0
STAND = 1

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
    dealer_card = pick_card()
    initial_state = (player_sum, usable_ace, dealer_card)
    return initial_state


def player_policy(state):
    player_sum, usable_ace, dealer_card = state
    action = STAND if player_sum >= 15 else HIT
    return action


def dealer_policy(dealer_sum):
    action = STAND if dealer_sum >= 17 else HIT
    return action


def player_turn(state):
    player_sum, usable_ace, dealer_card = state
    while True:
        action = player_policy(state)
        if action == STAND:
            break
        player_card = pick_card()
        player_sum, usable_ace = evaluate_hand(player_sum, usable_ace, player_card)
        state = (player_sum, usable_ace, dealer_card)
    return player_sum


def dealer_turn(card):
    usable_ace = False
    if card == 'A':
        usable_ace = True
    dealer_sum = evaluate_card(card)
    while True:
        card = pick_card()
        dealer_sum, usable_ace = evaluate_hand(dealer_sum, usable_ace, card)
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
    player_sum = player_turn(initial_state)
    if player_sum > 21:
        return -1
    dealer_card = initial_state[2]
    dealer_sum = dealer_turn(dealer_card)
    if dealer_sum > 21:
        return 1
    return game_result(player_sum, dealer_sum)


def run_simulation():
    results = []
    for i in range(100000):
        result = play_game()
        results.append(result)
    print(np.mean(results))


if __name__ == '__main__':
    run_simulation()
