ROWS = 3
COLUMNS = 3
MAX = 'O'
MIN = 'X'
MAX_WIN = 100
MAX_LOSE = -100
DRAW = 0
INITIAL_SPACE = "       "
MAX_DEPTH = 2

#import sys
#sys.stdout = open('output.txt', 'w')

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
        self.chosen_move = None

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
    def check_rows(self, index = None):
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
                if (index != None):
                    index.append(i)
                return True 
        return False 

    # utility for checking if somebody has won
    def check_cols(self, index = None):
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
                if (index != None):
                    index.append(j)
                return True 
        return False 

    # utility for checking if somebody has won
    def check_diags(self, index = None):
        player = self.current_state[0][0]
        if (player != ''):
            flag = True 
            for i in range(1, ROWS):
                if (self.current_state[i][i] != player):
                    flag = False
                    break
            if (flag == True):
                if (index != None):
                    index.append(1)
                return True 
        player = self.current_state[0][COLUMNS - 1]
        if (player != ''):
            flag = True
            for i, j in zip(range(1, ROWS), range(COLUMNS - 2, -1, -1)):
                if (self.current_state[i][j] != player):
                    flag = False
                    break
            if (flag == True):
                if (index != None):
                    index.append(2)
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

    # get the winning blocks
    def get_winning_blocks(self):
        index = []
        blocks = []
        if (self.check_rows(index)):
            for i in range(COLUMNS):
                blocks.append(str(index[0]) + str(i))
        elif (self.check_cols(index)):
            for i in range(ROWS):
                blocks.append(str(i) + str(index[0]))
        elif (self.check_diags(index)):
            if (index[0] == 1):
                for i in range(ROWS):
                    blocks.append(str(i) + str(i))
            else:
                for i, j in zip(range(ROWS), range(COLUMNS - 1, -1, -1)):
                    blocks.append(str(i) + str(j))

        return blocks

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

    # reinitialize game
    def restart_game(self, active_player):
        self.current_state = INITIAL_GAME_STATE
        self.active_player = active_player
        self.final_score = 0
        self.chosen_move = None

    def count_possible_wins(self):
        max_count = 0
        min_count = 0
        rows = []
        # rows
        for i in range(ROWS):
            row = []
            for j in range(COLUMNS):
                row.append(self.current_state[i][j])
            rows.append(list(row))

        # columns
        for j in range(COLUMNS):
            row = []
            for i in range(ROWS):
                row.append(self.current_state[i][j])
            rows.append(list(row))

        # diagonals
        row = []
        for i, j in zip(range(ROWS), range(COLUMNS - 1, -1, -1)):
            row.append(self.current_state[i][j])
        rows.append(list(row))

        row = []
        for i in range(ROWS):
            row.append(self.current_state[i][i])
        rows.append(list(row))

        for row in rows:
            if (MAX in row and MIN in row):
                pass
            elif (MAX in row):
                temp = row.count(MAX)
                if (temp == 3):
                    max_count += 100
                elif (temp == 2):
                    max_count += 10
                else:
                    max_count += 1
            elif (MIN in row):
                temp = row.count(MIN)
                if (temp == 3):
                    min_count += 100
                elif (temp == 2):
                    min_count += 10
                else:
                    min_count += 1
            else:
                max_count += 1
                min_count += 1

        return max_count, min_count

    def heuristic_score(self):
        max_ways, min_ways = self.count_possible_wins()
        return max_ways - min_ways
        

# Alpha Beta pruning
def alphabeta(game, alpha, beta, depth):
    if (game.is_over()):
        return game.score(depth)
    if (depth == MAX_DEPTH):
        return game.heuristic_score()

    depth += 1
    moves = game.get_available_moves()
    current_player = game.active_player
    current_state = game.current_state
    v = None
    for move in moves:
        possible_state = game.get_state(move)
        game.switch_player()
        game.current_state = possible_state
        if (current_player == MAX):
            v = -float("inf")
            v = alphabeta(game, alpha, beta, depth)
            alpha = max(alpha, v)
            if (v >= beta):
                break
        else:
            v = float("inf") 
            v = alphabeta(game, alpha, beta, depth)
            beta = min(v, beta)
            if (v <= alpha):
                break
        game.current_state = current_state
        game.active_player = current_player
    if (current_player == MAX):
        return alpha
    else:
        return beta
        
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
        val = alphabeta(game, -float("inf"), float("inf"), depth)
        scores.append(val)
        game.current_state = current_state
        game.active_player = current_player
        
    # generate score based on who is the current player
    game.active_player = current_player
    if (game.active_player == MAX):
        max_score = -float("inf")
        for i in range(len(scores)):
            if (scores[i] > max_score):
                max_score_index = i
                max_score = scores[i]
        game.chosen_move = moves[max_score_index]
        state = game.get_state(moves[max_score_index])
        game.current_state = state
        return max_score
 
    if (game.active_player == MIN):
        min_score = float("inf")
        for i in range(len(scores)):
            if (scores[i] < min_score):
                min_score_index = i
                min_score = scores[i]
        game.chosen_move = moves[min_score_index]
        state = game.get_state(moves[min_score_index])
        game.current_state = state
        return min_score


"""
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
    """
