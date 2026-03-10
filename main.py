from src.mcts.mcts import *

own, opp = init_board
player = 1
with timed(timer, 'legal'):
    a = get_legal_board(0, 0)

print(a)
timer.report()

