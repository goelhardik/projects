import config
import deck

class Player():

    def __init__(self, name = ''):
        self.name = name
        self.cards = []
        self.score = 0
        self.status = config.INPLAY
        self.last_choice = None

    def print_info(self):
        print(self.name)
        print("Score = " + str(self.score))
        for card in self.cards:
            card.print_card()

    def get_name(self):
        return self.name

    def get_status(self):
        return self.status

    def set_status(self, status):
        self.status = status
        return

    def get_score(self):
        return self.score

    def get_last_choice(self):
        return self.last_choice

    def update_state(self, choice, card = None):
        if (choice == config.STAY):
            self.last_choice = config.STAY
        elif (choice == config.HIT):
            self.last_choice = config.HIT
            self.cards.append(card)
            
            # update new score
            ace_flag = 0
            score = 0
            for card in self.cards:
                if (card.value in config.NUMBER_CARDS):
                    score += int(card.value)
                elif (card.value == 'A'):
                    ace_flag = 1
                    score += 1
                else:
                    score += 10

            alt_score = float("inf")
            if (ace_flag == 1):
                alt_score = score - 1 + 11

            if (alt_score <= config.MAX_SCORE):
                self.score = alt_score
            else:
                self.score = score

            if (self.score == config.MAX_SCORE):
                self.status = config.HOME
            elif (self.score > config.MAX_SCORE):
                self.status = config.BUSTED

class Game():

    def __init__(self, p1 = None, p2 = None):
        self.p1 = p1
        self.p2 = p2
        self.deck = deck.Deck()
        self.state = config.BEGIN
        self.active_player = self.p1
        self.inactive_player = self.p2

    def print_game_info(self):
        print("#####################")
        print("Game info:")
        self.p1.print_info()
        self.p2.print_info()
        print("#####################")
        
    def is_over(self):
        if (self.deck.cards_remaining() == 0):
            return True
        if (self.p1.get_status() != config.INPLAY and self.p2.get_status() != 
            config.INPLAY):
            return True

    def print_final_state(self):
        print(self.p1.get_name(), end = " ")
        if (self.p1.get_status() == config.BUSTED):
            print("BUSTED")
        else:
            print("score is " + str(self.p1.get_score()))

        print(self.p2.get_name(), end = " ")
        if (self.p2.get_status() == config.BUSTED):
            print("BUSTED")
        else:
            print("score is " + str(self.p2.get_score()))

        return

    def switch_players(self):
        self.active_player, self.inactive_player = self.inactive_player, self.active_player
        return

    def start_play(self):

        # initialize game stuff
        print("Game beginning!")
        self.p1 = Player()
        self.p2 = Player()
        self.p1.name = str(input("Enter player 1 name: "))
        self.p2.name = str(input("Enter player 2 name: "))
        self.active_player = self.p1
        self.inactive_player = self.p2

        # start playing
        while (1):
            self.print_game_info()
            if (self.is_over()):
                print("Game is over!")
                break

            if (self.active_player.get_status() != config.INPLAY):
                print(self.active_player.get_name() + " is done!")
                self.switch_players()
                continue

            choice = str(input(self.active_player.get_name() + " to choose (Hit : 'h' or Stay : 's') : "))
            if (choice == 's'):
                if (self.active_player.get_last_choice() == config.STAY):
                    print(self.active_player.get_name() + " chose to stay again, so done!")
                    self.active_player.set_status(config.HOME)
                    self.switch_players()
                    continue
                self.active_player.update_state(config.STAY)
                if (self.inactive_player.get_last_choice() == config.STAY):
                    print("Both players stayed")
                    self.print_game_info()
                    break
                else:
                    self.switch_players()
            elif (choice == 'h'):
                card = self.deck.draw_card()
                card.print_card()
                self.active_player.update_state(config.HIT, card)
                self.switch_players()
            else:
                print("Invalid choice, try again : ")

        self.print_final_state()


g = Game()
g.start_play()
