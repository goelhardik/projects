A naive implementation of the game of Blackjack.

This is a two-player implementation. They are just playing solo. The dealer is 
not playing. In the end, either they get busted, or they get a score.
This is done just as a programming practice project. It is not meant to be a 
real model of Blackjack.

Each player can choose to either hit or stay. If hit is chosen, a card gets 
drawn from the dealer's deck and gets added to the player's score.

If a player chooses to stay twice, then the game is over for them.
If both the players choose to stay, the game is over.
If both players are busted, the game is over.

Number cards have a value same as the number.
All face cards have a value of 10.
Ace has a value of 1 or 11 depending on what is good for the player.
Going above 21 busts the player.

###############################

How to run:

Need python3.4.3 to execute this.

Just run the following command:
python game.py
