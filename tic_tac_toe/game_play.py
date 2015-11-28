#!/usr/bin/python
BOARD_COLOR = "white"

from tkinter import *
from backend import *

state = INITIAL_GAME_STATE
game = Game(state)
game.active_player = MAX

root = Tk()
root.geometry("400x300")
root.title("Tic Tac Toe")
root.configure(background = BOARD_COLOR)

final_msg = Text(root, height = 1)
final_msg.place(x = 3, y = 50)

def Enter(i, j, player):
    button = buttons[i][j]
    button.config(text = player)
    if (player == MIN):
        color = "OliveDrab1"
    else:
        color = "brown1"
    button.config(background = BOARD_COLOR, activebackground = BOARD_COLOR, state = 
                 DISABLED, foreground = color)

    if (game.is_over()):
        if (game.check_win()):
            final_msg.insert(END, "You Lose!")
        else:
            final_msg.insert(END, "Game Draw!")

    if (player == MIN):
        user_move = Move(i, j, MIN)
        user_state = game.get_state(user_move)
        game.current_state = user_state
        play(game, 0)
        computer_move = game.chosen_move
        Enter(computer_move.i, computer_move.j, computer_move.mark)

board = Frame(root, height = 128, width = 128)
board.configure(background = BOARD_COLOR)

def restart_game():
    game.current_state = INITIAL_GAME_STATE
    game.active_player = MAX
    initialize_game()

restart_button = Button(root, text = "Restart", command = restart_game)

def create_function(i, j):
    return lambda : Enter(i, j, MIN)

buttons = [[None for i in range(ROWS)] for j in range(COLUMNS)]
def initialize_game():
    final_msg.delete('1.0', END)
    for i in range(ROWS):
        for j in range(COLUMNS):
            buttons[i][j] = Button(board, background = "LightPink1", activebackground = 
                               "LightPink2", height = 3, width = 5, bd = 4, text = "  ",  command = 
                               create_function(i, j))
            buttons[i][j].grid(row = i, column = j)

initialize_game()
restart_button.pack(side = RIGHT)
board.place(x = 100, y = 100)

root.mainloop()
