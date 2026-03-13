from src.mcts.mcts import *

own, opp = init_board
x = get_legal_board(own, opp)
y = bitboard_to_array(x)
print(y)
print(np.random.choice(y))
breakpoint()
player = 1
pass_count = 0
action = None

mcts = MCTS(default_model, max_nodes=100000)

while True:
    pi = mcts.search(own, opp, last_action=action, timer=timer)
    action = select_action_from_pi(pi, temperature=1.0)

    pass_count = pass_count + 1 if action == PASS_ACTION else 0
    if pass_count == 2:
        break

    own, opp = apply_move_bitboard(own, opp, action)
    own, opp = opp, own
    player = -player

