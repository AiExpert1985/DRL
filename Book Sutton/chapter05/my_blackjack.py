import numpy as np

DECK = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'] * 4
CARD_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
               'J': 10, 'Q': 10, 'K': 10}

def pick_card():
    return np.random.choice(DECK)

def card_value(card):
    return CARD_VALUES[card]


def initialize_game():
    player_cards = [pick_card() for _ in range(2)]
    dealer_card = pick_card()
    return player_cards, dealer_card


def play():
    pass

def run():
    player_cards, dealer_card = initialize_game()
    print(player_cards, dealer_card)


if __name__ == '__main__':
    run()
