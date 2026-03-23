
import threading
from typing import Optional

import numpy as np
import torch

from mcts import (
    OthelloResNet,
    select_ucb, backup_path, backup_path_batch,
    get_or_create_child, apply_move_bitboard,
    get_legal_board, bitboard_to_array,
    bitboard_to_input, popcount,
    ACTION_SIZE, PASS_ACTION, MAX_DEPTH, MAX_NODE,
    MCTS_SIMS, BATCH_SIZE, DEVICE, default_model, timed, timer, init_board
)


"""
MCTS with:
  1. Virtual loss + parallel rollouts  (threading, N_WORKERS threads)
  2. Deep tree reuse via subtree compaction  (preserves full subtree on move)

Drop-in replacement for the original MCTS class.
All Numba-compiled helpers (select_ucb, backup_path, backup_path_batch,
get_or_create_child, apply_move_bitboard, get_legal_board, bitboard_to_array,
bitboard_to_input, popcount) are assumed to exist unchanged.
"""

N_WORKERS = 4
VIRTUAL_LOSS = 3




if __name__ == "__main__":
    mcts = MCTS(default_model)
    own, opp = init_board()
    with timed(timer, "parallel"):
        pi = mcts.search(own, opp)

    timer.report()
    print(pi)
