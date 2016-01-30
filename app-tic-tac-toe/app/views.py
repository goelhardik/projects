from flask import render_template, flash, redirect, request, jsonify
from app import app
from game import backend 
import json

def make_game_from_state(state):
    current_state = [['' for j in range(backend.COLUMNS)] for i in 
                     range(backend.ROWS)]
    for i in range(backend.ROWS):
        for j in range(backend.COLUMNS):
            if (state[i * backend.COLUMNS + j] != '-'):
                current_state[i][j] = state[i * backend.COLUMNS + j]

    game = backend.Game(current_state)
    return game

@app.route('/get_winning_blocks')
def get_winning_blocks():
    state = request.args.get('gamestate', '00', type = str)
    game = make_game_from_state(state)
    blocks = game.get_winning_blocks()
    return json.dumps(blocks)

@app.route('/restart_game')
def restart_game():
    return jsonify(result = "0")

@app.route('/check_if_over')
def check_if_over():
    state = request.args.get('gamestate', '00', type = str)
    game = make_game_from_state(state)
    message = "-1"
    if (game.is_over()):
        if (game.check_win()):
            message = "1"
        else:
            message = "0"
    return jsonify(result = message)
            

@app.route('/game_play')
def game_play():
    state = request.args.get('gamestate', '00', type = str)
    game = make_game_from_state(state)
    if (game.is_over()):
        return jsonify(result = "")
    backend.play(game, 0)
    computer_move = game.chosen_move
    return jsonify(result = (str(computer_move.i) + str(computer_move.j)))

@app.route('/')
def base():
    return render_template('base.html', title = 'Base')
