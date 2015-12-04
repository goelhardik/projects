DIM = 3
BOARDS = 3
ROWS = 3
COLUMNS = 3
MAX = 'O'
MIN = 'X'
MAX_WIN = 100 
MAX_LOSE = -100
DRAW = 0
INITIAL_SPACE = "       "

MAX_DEPTH = 4

# constants for checking game wins
BOARD_WISE = 0
ROW_WISE = 1
COLUMN_WISE = 2

POSSIBLE_WINS = [
                ['000', '001', '002'],
                ['010', '011', '012'],
                ['020', '021', '022'],
                ['000', '010', '020'],
                ['001', '011', '021'],
                ['002', '012', '022'],
                ['000', '011', '022'],
                ['020', '011', '002'],

                ['100', '101', '102'],
                ['110', '111', '112'],
                ['120', '121', '122'],
                ['100', '110', '120'],
                ['101', '111', '121'],
                ['102', '112', '122'],
                ['100', '111', '122'],
                ['120', '111', '102'],

                ['200', '201', '202'],
                ['210', '211', '212'],
                ['220', '221', '222'],
                ['200', '210', '220'],
                ['201', '211', '221'],
                ['202', '212', '222'],
                ['200', '211', '222'],
                ['220', '211', '202'],

                ['000', '100', '200'],
                ['001', '101', '201'],
                ['002', '102', '202'],
                ['010', '110', '210'],
                ['011', '111', '211'],
                ['012', '112', '212'],
                ['020', '120', '220'],
                ['021', '121', '221'],
                ['022', '122', '222'],

                ['000', '101', '202'],
                ['200', '101', '002'],
                ['010', '111', '212'],
                ['210', '111', '012'],
                ['020', '121', '222'],
                ['220', '121', '022'],
                ['000', '110', '220'],
                ['200', '110', '020'],
                ['001', '111', '221'],
                ['201', '111', '021'],
                ['002', '112', '222'],
                ['202', '112', '022'],
                ['000', '111', '222'],
                ['200', '111', '022'],
                ['020', '111', '202'],
                ['002', '111', '220']
                ]
#import sys
#sys.stdout = open('output.txt', 'w')

from pprint import pprint

class Game:
    
    def __init__(self, current_state = {}, active_player = MAX):
        if (len(current_state) == 0):
            self.current_state = self.make_initial_state() 
            #self.current_state = self.make_rigged_state() 
        else:
            self.current_state = current_state
        self.active_player = active_player
        self.final_score = 0
        self.chosen_move = None

    def get_boxid(self, k, i, j):
        return (str(k) + str(i) + str(j))

    def make_initial_state(self):
        state = {}
        for k in range(DIM):
            for i in range(DIM):
                for j in range(DIM):
                    state[self.get_boxid(k, i, j)] = "-"

        return state

    def make_rigged_state(self):
        state = self.make_initial_state()
        state['000'] = MAX
        state['002'] = MAX
        state['020'] = MAX
        state['021'] = MAX
        state['111'] = MAX
        state['010'] = MIN
        state['011'] = MIN
        state['022'] = MIN
        state['102'] = MIN
        return state

    # get all available moves for the active_player in the current_state
    def get_available_moves(self):
        moves = []
        for key in self.current_state.keys():
            if (self.current_state[key] == "-"):
                move = {key : self.active_player}
                # move = Move(key, self.active_player)
                moves.append(move.copy())

        return moves

    # get a game state after the passed move is made
    def get_state(self, move):
        new_state = self.current_state.copy() 
        key, value = move.popitem()
        new_state[key] = value
        return new_state

    # check if the game is over
    def is_over(self):
        is_win, winner, candidates = self.check_win()
        if (is_win):
            return True
        if ("-" in self.current_state.values()):
            return False
        else:
            return True
        
    # get if there is a winner from the candidate box list
    def check_and_get_winner(self, candidate_boxids):
        result = {}
        for candidate_id in candidate_boxids:
            mark = self.current_state[candidate_id]
            result[mark] = mark 

        if ("-" in result.keys() or len(result) > 1):
            return "-"
        else:
            boxid, player = result.popitem()
            return player

    # check for any wins on a 1D board; here board contains DIM * DIM boxids
    def check_1D_win(self, board):
        for i in range(DIM):
            candidates = []
            for j in range(i * DIM, i * DIM + DIM):
                candidates.append(board[j])
            winner = self.check_and_get_winner(candidates)
            if (winner == "-"):
                pass
            else:
                return True, winner, candidates

        for i in range(DIM):
            candidates = []
            for j in range(i, len(board), DIM):
                candidates.append(board[j])
            winner = self.check_and_get_winner(candidates)
            if (winner == "-"):
                pass
            else:
                return True, winner, candidates

        candidates = []
        for i in range(DIM):
            candidates.append(board[i * (DIM + 1)])
        winner = self.check_and_get_winner(candidates)
        if (winner == "-"):
            pass
        else:
            return True, winner, candidates

        candidates = []
        for i in range(DIM):
            candidates.append(board[(i + 1) * (DIM - 1)])
        winner = self.check_and_get_winner(candidates)
        if (winner == "-"):
            pass
        else:
            return True, winner, candidates

        # no winner
        return False, None, None

    # construct 1D board and check
    def check_win_1d_ttt(self, num, which_wise):
        board = []
        for i in range(DIM):
            for j in range(DIM):
                board.append(str(i) + str(j))
        for k in range(len(board)):
            board[k] = board[k][: which_wise] + str(num) + board[k][which_wise :]
        is_win, winner, boxes = self.check_1D_win(board)
        return is_win, winner, boxes

    # check if somebody has won
    def check_win(self):
        # board-wise 1D ttts
        for k in range(DIM):
            board_win, player, candidates = self.check_win_1d_ttt(k, BOARD_WISE)
            if (board_win):
                return True, player, candidates
        # row-wise 1D ttts
        for k in range(DIM):
            board_win, player, candidates = self.check_win_1d_ttt(k, ROW_WISE)
            if (board_win):
                return True, player, candidates
        # column-wise 1D ttts
        for k in range(DIM):
            board_win, player, candidates = self.check_win_1d_ttt(k, COLUMN_WISE)
            if (board_win):
                return True, player, candidates
        # 4 3D diagonals; HARDCODED
        candidates = ["000", "111", "222"]
        winner = self.check_and_get_winner(candidates)
        if (winner == "-"):
            pass
        else:
            return True, winner, candidates
        candidates = ["002", "111", "220"]
        winner = self.check_and_get_winner(candidates)
        if (winner == "-"):
            pass
        else:
            return True, winner, candidates
        candidates = ["200", "111", "022"]
        winner = self.check_and_get_winner(candidates)
        if (winner == "-"):
            pass
        else:
            return True, winner, candidates
        candidates = ["202", "111", "020"]
        winner = self.check_and_get_winner(candidates)
        if (winner == "-"):
            pass
        else:
            return True, winner, candidates

        # no winner
        return False, None, None
    
    # return the score based on the result of the game
    def score(self, depth):
        self.switch_player()
        is_win, winner, candidates = self.check_win()
        if (is_win):
            if (winner == MAX):
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
        pprint(self.current_state)
        
    # reinitialize game
    def restart_game(self, active_player):
        self.current_state = self.make_intial_state()
        self.active_player = active_player
        self.final_score = 0
        self.chosen_move = None

    def count_possible_wins(self):
        rows = []
        for row in POSSIBLE_WINS:
            row_players = []
            for key in row:
                row_players.append(self.current_state[key])
            rows.append(list(row_players))

        max_count = 0
        min_count = 0
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
        possible_state = game.get_state(move.copy())
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
        possible_state = game.get_state(move.copy())
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
        state = game.get_state(moves[max_score_index].copy())
        game.current_state = state
        return max_score
 
    if (game.active_player == MIN):
        min_score = float("inf")
        for i in range(len(scores)):
            if (scores[i] < min_score):
                min_score_index = i
                min_score = scores[i]
        game.chosen_move = moves[min_score_index]
        state = game.get_state(moves[min_score_index].copy())
        game.current_state = state
        return min_score


game = Game()
game.active_player = MAX
game.print_state()
while (not game.is_over()):
    user_move = input('Enter your choice : ')
    user_state = game.get_state({user_move : MIN})
    game.current_state = user_state
    print("Your input :")
    game.print_state()
    play(game, 0)
    print("Computer's turn :")
    game.print_state()
is_win, winner, candidates = game.check_win()
if (not is_win):
    print("Game Drawn!!")
else:
    print("Winner is " + winner)
    print(candidates)
