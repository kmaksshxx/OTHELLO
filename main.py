from src.mcts.mcts import *


own, opp = init_board
player = 1
mcts = MCTS(default_model)
timer.reset('MCTS')
action = None
pass_count = 0


while True:
    pi = mcts.search(own, opp, last_action=action, timer=timer)

    action = select_action_from_pi(pi, 0.01)

    pass_count = pass_count + 1 if action == PASS_ACTION else 0
    if pass_count == 2:
        break

    own, opp = apply_move_bitboard(own, opp, action)
    own, opp = opp, own
    player = -player

timer.report()
render(own, opp, player)

