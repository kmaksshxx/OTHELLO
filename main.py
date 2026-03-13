from src.mcts.mcts import *

own, opp = init_board()

player = 1
pass_count = 0

while True:
    action = get_random_action(own, opp)
    own, opp = apply_move_bitboard(own, opp, action)
    own, opp = opp, own
    player = -player

    pass_count = pass_count + 1 if action == PASS_ACTION else 0
    if pass_count == 2:
        break

diff = popcount(own) - popcount(opp)
winner = player if diff > 0 else -player if diff < 0 else 0
print(winner)
render(own, opp, player)



