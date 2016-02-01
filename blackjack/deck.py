import config
import random

class Card():

    def __init__(self, value, suit):
        self.suit = suit
        self.value = value

    def print_card(self):
        print(str(self.value) + " of " + config.SUIT_NAME(self.suit))

class Suit():

    def __init__(self, suit):
        self.cards = []
        for value in config.NUMBER_CARDS:
            self.cards.append(Card(value, suit))
        for value in config.FACE_CARDS:
            self.cards.append(Card(value, suit))

        self.num_cards = len(self.cards)

class Deck():

    def __init__(self):
        self.num_cards = config.TOTAL_CARDS 
        self.cards = [Suit(suit_type) for suit_type in config.SUIT_TYPES]

    def cards_remaining(self):
        return self.num_cards

    def draw_card(self):
        if (self.num_cards == 0):
            print("Card Deck empty!")
            return -1

        # draw a random number in remaining cards and map it to a card
        rand_card = random.randrange(self.num_cards)
        for suit in range(config.NUM_SUITS):
            if (self.cards[suit].num_cards > rand_card):
                chosen_suit = suit
                break
            else:
                rand_card -= self.cards[suit].num_cards

        # now find rand_cardth card in suit chosen_suit
        chosen_card = self.cards[chosen_suit].cards[rand_card]
        
        # delete the card from the deck
        self.num_cards -= 1
        self.cards[chosen_suit].num_cards -= 1
        del self.cards[chosen_suit].cards[rand_card]

        return chosen_card

    def  print_deck(self):
        print("##################################")
        for suit in range(config.NUM_SUITS):
            print("SUIT = " + str(suit))
            for card in range(self.cards[suit].num_cards):
                print(self.cards[suit].cards[card].value, end = " ")
            print()
        print("##################################")
