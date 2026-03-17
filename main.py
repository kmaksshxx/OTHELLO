from src.train.train import *

ck = load_checkpoint()

default_model.load_state_dict(ck['model'])

mcts = MCTS(default_model, n_sim=50)

own, opp = init_board()
player = 1
pass_count = 1
action = None

while True:
    pi = mcts.search(own, opp, last_action=action)
    action = select_action_from_pi(pi, 1.0)
    inp = bitboard_to_input(own, opp)
    inp_t = torch.from_numpy(inp).unsqueeze(0)

    x, y = default_model(inp_t)
    print(y.squeeze())

    own, opp = apply_move_bitboard(own, opp, action)
    own, opp = opp, own
    player = -player

    pass_count = pass_count + 1 if action == PASS_ACTION else 0
    if pass_count == 2:
        break




