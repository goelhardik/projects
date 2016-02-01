# game states
BEGIN = 0
IN_PROG = 1
OVER = 2

# player states
INPLAY = 0
BUSTED = 1
HOME = 2

# player choice
HIT = 0
STAY = 1

MAX_SCORE = 21

###############################
# specifying the deck details #
###############################

# card suits
SPADES = 0
CLUBS = 1
HEARTS = 2
DIAMONDS = 3
SUIT_TYPES = [SPADES, CLUBS, HEARTS, DIAMONDS]
SUIT_NAMES = ['Spades', 'Clubs', 'Hearts', 'Diamonds']
NUM_SUITS = 4

def SUIT_NAME(suit):
    return SUIT_NAMES[suit]


# card types
NUMBER_CARDS = ['2', '3', '4', '5', '6', '7', '8', '9', '10']
FACE_CARDS = ['J', 'Q', 'K', 'A']

# total cards
TOTAL_CARDS = 52
SUIT_CARDS = TOTAL_CARDS // NUM_SUITS


def enum(args):
    enums = dict(zip(args, range(len(args))))
    return enums 
