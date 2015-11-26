ROWS = 3
COLUMNS = 3
MAX = 'O'
MIN = 'X'
MAX_WIN = 10
MAX_LOSE = -10
DRAW = 0
INITIAL_SPACE = "       "

class Move:

    def __init__(self, i, j, player):
        self.i = i
        self.j = j
        self.mark = player

INITIAL_GAME_STATE = [ ['', '', ''],
                       ['', '', ''],
                       ['', '', '']
                     ]
class Game:
    
    def __init__(self, current_state = INITIAL_GAME_STATE, active_player = MAX):
        self.current_state = current_state
        self.active_player = active_player
        self.final_score = 0

    # get all available moves for the active_player in the current_state
    def get_available_moves(self):
        moves = []
        for i in range(ROWS):
            for j in range(COLUMNS):
                if (self.current_state[i][j] == ''):
                    move = Move(i, j, self.active_player)
                    moves.append(move)

        return moves

    # return a copy of the given state
    def copy_game_state(self, given_state):
        new_state = [['' for i in range(ROWS)] for j in range(COLUMNS)]
        for i in range(ROWS):
            for j in range(COLUMNS):
                new_state[i][j] = given_state[i][j]

        return new_state

    # get a game state after the passed move is made
    def get_state(self, move):
        new_state = self.copy_game_state(self.current_state)
        new_state[move.i][move.j] = move.mark
        return new_state

    # check if the game is over
    def is_over(self):
        if (self.check_win()):
            return True
        for i in range(ROWS):
            for j in range(COLUMNS):
                if (self.current_state[i][j] == ''):
                    return False
        return True
        
    # utility for checking if somebody has won
    def check_rows(self):
        for i in range(ROWS):
            flag = True 
            player = self.current_state[i][0]
            if (player == ''):
                continue
            for j in range(1, COLUMNS):
                if (self.current_state[i][j] != player):
                    flag = False
                    break
            if (flag == True):
                return True
        return False

    # utility for checking if somebody has won
    def check_cols(self):
        for j in range(COLUMNS):
            flag = True 
            player = self.current_state[0][j]
            if (player == ''):
                continue
            for i in range(1, ROWS):
                if (self.current_state[i][j] != player):
                    flag = False
                    break
            if (flag == True):
                return True
        return False

    # utility for checking if somebody has won
    def check_diags(self):
        player = self.current_state[0][0]
        if (player != ''):
            flag = True 
            for i in range(1, ROWS):
                if (self.current_state[i][i] != player):
                    flag = False
                    break
            if (flag == True):
                return True
        player = self.current_state[0][COLUMNS - 1]
        if (player != ''):
            flag = True
            for i, j in zip(range(1, ROWS), range(COLUMNS - 2, -1, -1)):
                if (self.current_state[i][j] != player):
                    flag = False
                    break
            if (flag == True):
                return True
        return False

    # check if somebody has won
    def check_win(self):
        row = self.check_rows()
        col = self.check_cols()
        diag = self.check_diags()
        if (row or col or diag):
            return True
        else:
            return False

    # return the score based on the result of the game
    def score(self, depth):
        self.switch_player()
        if (self.check_win()):
            if (self.active_player == MAX):
                return MAX_WIN - depth
            else:
                return MAX_LOSE - depth
        else:
            return DRAW

    # switch the active player
    def switch_player(self):
        if (self.active_player == MAX):
            self.active_player = MIN
        else:
            self.active_player = MAX

    # print the current state of the board
    def print_state(self):
        print("\n" + INITIAL_SPACE + "   1  2  3\n")
        for i in range(ROWS):
            print(INITIAL_SPACE + str(i + 1), end = "  ")
            for j in range(COLUMNS):
                char = self.current_state[i][j]
                if (char == ''):
                    char = '-'
                print(char, end = "  ")
            print("\n")

# play the game using the minimax algorithm
def play(game, depth):
    # if game is over, return the score
    if (game.is_over()):
        return game.score(depth)

    depth += 1
    # get scores for each possible move using minimax
    scores = []
    moves = game.get_available_moves()
    current_player = game.active_player
    current_state = game.current_state
    for move in moves:
        possible_state = game.get_state(move)
        game.switch_player()
        game.current_state = possible_state
        scores.append(play(game, depth))
        game.current_state = current_state
        game.switch_player()
        
    # generate score based on who is the current player
    game.active_player = current_player
    if (game.active_player == MAX):
        max_score = -float("inf")
        for i in range(len(scores)):
            if (scores[i] > max_score):
                max_score_index = i
                max_score = scores[i]
        state = game.get_state(moves[max_score_index])
        game.current_state = state
        return max_score
 
    if (game.active_player == MIN):
        min_score = float("inf")
        for i in range(len(scores)):
            if (scores[i] < min_score):
                min_score_index = i
                min_score = scores[i]
        state = game.get_state(moves[min_score_index])
        game.current_state = state
        return min_score

state = INITIAL_GAME_STATE
game = Game(state)
game.active_player = MAX
game.print_state()
while (not game.is_over()):
    row, column = [int(x) - 1 for x in
                   input('Enter your choice (row, column) : ').split(',')]
    user_move = Move(row, column, MIN)
    user_state = game.get_state(user_move)
    game.current_state = user_state
    print("Your input :")
    game.print_state()
    play(game, 0)
    print("Computer's turn :")
    game.print_state()

if (not game.check_win()):
    print("Game Drawn!!")
else:
    print("You lose!")
