import numpy as np
import numba as nb
from typing import Tuple
from collections import defaultdict
from contextlib import contextmanager

import time
import matplotlib.pyplot as plt
import torch.cuda

BOARD_SIZE = 8
PASS_ACTION = BOARD_SIZE ** 2
ACTION_SIZE = PASS_ACTION + 1
init_board = (34628173824, 68853694464)
PASS_MOVE = np.uint64(64)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Masks to prevent wrap-around when shifting horizontally/diagonally
NOT_A_FILE = np.uint64(0xfefefefefefefefe)  # Clears bits in column A (0)
NOT_H_FILE = np.uint64(0x7f7f7f7f7f7f7f7f)  # Clears bits in column H (7)


@nb.njit(nb.uint64(nb.uint64, nb.uint64))
def get_legal_board(own, opp) -> int:
    blank = ~(own | opp)
    legal_moves = np.uint64(0)

    # 1. 좌 (West)
    t = ((own & NOT_A_FILE) >> 1) & opp  # Shift own left, ensure no wrap from col 0 to 7 of prev row
    for _ in range(5): t |= (((t & NOT_A_FILE) >> 1) & opp)
    legal_moves |= (((t & NOT_A_FILE) >> 1) & blank)

    # 2. 우 (East)
    t = ((own & NOT_H_FILE) << 1) & opp  # Shift own right, ensure no wrap from col 7 to 0 of prev row
    for _ in range(5): t |= (((t & NOT_H_FILE) << 1) & opp)
    legal_moves |= (((t & NOT_H_FILE) << 1) & blank)

    # 3. 상 (North)
    t = (own >> 8) & opp
    for _ in range(5): t |= ((t >> 8) & opp)
    legal_moves |= ((t >> 8) & blank)

    # 4. 하 (South)
    t = (own << 8) & opp
    for _ in range(5): t |= ((t << 8) & opp)
    legal_moves |= ((t << 8) & blank)

    # 5. 좌상 (North West)
    t = ((own & NOT_A_FILE) >> 9) & opp
    for _ in range(5): t |= (((t & NOT_A_FILE) >> 9) & opp)
    legal_moves |= (((t & NOT_A_FILE) >> 9) & blank)

    # 6. 우상 (North East)
    t = ((own & NOT_H_FILE) >> 7) & opp
    for _ in range(5): t |= (((t & NOT_H_FILE) >> 7) & opp)
    legal_moves |= (((t & NOT_H_FILE) >> 7) & blank)  # Corrected from NOT_A_FILE

    # 7. 좌하 (South West)
    t = ((own & NOT_A_FILE) << 7) & opp
    for _ in range(5): t |= (((t & NOT_A_FILE) << 7) & opp)
    legal_moves |= (((t & NOT_A_FILE) << 7) & blank)

    # 8. 우하 (South East)
    t = ((own & NOT_H_FILE) << 9) & opp
    for _ in range(5): t |= (((t & NOT_H_FILE) << 9) & opp)
    legal_moves |= (((t & NOT_H_FILE) << 9) & blank)

    return legal_moves


@nb.njit(nb.types.Array(nb.int64, 1, 'C')(nb.uint64))
def bitboard_to_array(bitboard) -> np.ndarray:
    moves = np.empty(64, dtype=np.int64)  # Preallocate array for Numba compatibility
    count = 0
    i = 0
    while i < 64:
        if (bitboard >> np.uint64(i)) & np.uint64(1):
            moves[count] = i
            count += 1
        i += 1
    return moves[:count]  # Return only the filled part


@nb.njit
def board_to_bitboard(board: np.ndarray) -> Tuple[np.uint64, np.uint64]:
    """
    board: (8, 8) 크기의 np.ndarray (1: 흑, -1: 백)
    return: (black_bits, white_bits) -> (uint64, uint64)
    """
    black_bits = np.uint64(0)
    white_bits = np.uint64(0)

    # 1차원으로 flatten 하여 순회 (속도 최적화)
    flat_board = board.ravel()
    for i in range(64):
        if flat_board[i] == 1:
            black_bits |= (np.uint64(1) << np.uint64(i))
        elif flat_board[i] == -1:
            white_bits |= (np.uint64(1) << np.uint64(i))

    return black_bits, white_bits


@nb.njit
def bitboard_to_board(black_bits, white_bits) -> np.ndarray:
    """비트보드를 다시 np.ndarray로 복구 (디버깅/입력용)"""
    board = np.zeros(64, dtype=np.int8)
    for i in range(64):
        mask = np.uint64(1) << np.uint64(i)
        if black_bits & mask:
            board[i] = 1
        elif white_bits & mask:
            board[i] = -1
    return board.reshape((8, 8))


def board_to_input(board: np.ndarray, player: int) -> np.ndarray:
    """Return (1,2,8,8) tensor from board and player perspective"""
    planes = np.stack(
        [(board == player), (board == -player)], axis=0
    ).astype(np.float32)
    return np.expand_dims(planes, 0)


@nb.njit(nb.types.UniTuple(nb.uint64, 2)(nb.uint64, nb.uint64, nb.uint64))
def apply_move_bitboard(own, opp, action_idx) -> Tuple[int, int]:
    if action_idx == PASS_MOVE:
        return own, opp

    move = np.uint64(1) << action_idx
    flipped = np.uint64(0)

    # ---- West
    t = ((move & NOT_A_FILE) >> 1) & opp
    t |= (((t & NOT_A_FILE) >> 1) & opp)
    t |= (((t & NOT_A_FILE) >> 1) & opp)
    t |= (((t & NOT_A_FILE) >> 1) & opp)
    t |= (((t & NOT_A_FILE) >> 1) & opp)
    if ((t & NOT_A_FILE) >> 1) & own:
        flipped |= t

    # ---- East
    t = ((move & NOT_H_FILE) << 1) & opp
    t |= (((t & NOT_H_FILE) << 1) & opp)
    t |= (((t & NOT_H_FILE) << 1) & opp)
    t |= (((t & NOT_H_FILE) << 1) & opp)
    t |= (((t & NOT_H_FILE) << 1) & opp)
    if ((t & NOT_H_FILE) << 1) & own:
        flipped |= t

    # ---- North
    t = (move >> 8) & opp
    t |= ((t >> 8) & opp)
    t |= ((t >> 8) & opp)
    t |= ((t >> 8) & opp)
    t |= ((t >> 8) & opp)
    if (t >> 8) & own:
        flipped |= t

    # ---- South
    t = (move << 8) & opp
    t |= ((t << 8) & opp)
    t |= ((t << 8) & opp)
    t |= ((t << 8) & opp)
    t |= ((t << 8) & opp)
    if ((t << 8) & own):
        flipped |= t

    # ---- NW
    t = ((move & NOT_A_FILE) >> 9) & opp
    t |= (((t & NOT_A_FILE) >> 9) & opp)
    t |= (((t & NOT_A_FILE) >> 9) & opp)
    t |= (((t & NOT_A_FILE) >> 9) & opp)
    t |= (((t & NOT_A_FILE) >> 9) & opp)
    if (((t & NOT_A_FILE) >> 9) & own):
        flipped |= t

    # ---- NE
    t = ((move & NOT_H_FILE) >> 7) & opp
    t |= (((t & NOT_H_FILE) >> 7) & opp)
    t |= (((t & NOT_H_FILE) >> 7) & opp)
    t |= (((t & NOT_H_FILE) >> 7) & opp)
    t |= (((t & NOT_H_FILE) >> 7) & opp)
    if (((t & NOT_H_FILE) >> 7) & own):
        flipped |= t

    # ---- SW
    t = ((move & NOT_A_FILE) << 7) & opp
    t |= (((t & NOT_A_FILE) << 7) & opp)
    t |= (((t & NOT_A_FILE) << 7) & opp)
    t |= (((t & NOT_A_FILE) << 7) & opp)
    t |= (((t & NOT_A_FILE) << 7) & opp)
    if (((t & NOT_A_FILE) << 7) & own):
        flipped |= t

    # ---- SE
    t = ((move & NOT_H_FILE) << 9) & opp
    t |= (((t & NOT_H_FILE) << 9) & opp)
    t |= (((t & NOT_H_FILE) << 9) & opp)
    t |= (((t & NOT_H_FILE) << 9) & opp)
    t |= (((t & NOT_H_FILE) << 9) & opp)
    if (((t & NOT_H_FILE) << 9) & own):
        flipped |= t

    own ^= flipped | move
    opp ^= flipped

    return own, opp


@nb.njit(nb.float32[:, :, :](nb.uint64, nb.uint64), inline='always')
def bitboard_to_input(own, opp):
    inp = np.zeros((2, 8, 8), dtype=np.float32)

    for i in range(64):
        r = i >> 3
        c = i & 7

        inp[0, r, c] = (own >> i) & 1
        inp[1, r, c] = (opp >> i) & 1

    return inp


@nb.njit
def select_action_from_pi(pi: np.ndarray, temperature: float = 1.0) -> int:
    legal_idx = np.where(pi > 0)[0]
    legal_len = len(legal_idx)
    if legal_len == 0:
        return PASS_ACTION

    legal_pi = pi[legal_idx]

    if temperature == 0:
        # Find all indices that have the maximum value
        max_val = legal_pi.max()
        max_indices_in_legal_pi = np.where(legal_pi == max_val)[0]

        # Randomly select one index from those with the maximum value
        chosen_idx_in_legal_pi = np.random.choice(max_indices_in_legal_pi, size=1)[0]
        return int(legal_idx[chosen_idx_in_legal_pi])

    if temperature != 1.0:
        temp_f32 = np.float32(temperature)
        epsilon_f32 = np.float32(1e-20)
        logits = np.log(legal_pi + epsilon_f32) / temp_f32
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
    else:
        probs = legal_pi / legal_pi.sum()

    r = np.random.rand()
    cumulative_sum = 0.0
    selected_action_idx = -1
    for i in range(len(probs)):
        cumulative_sum += probs[i]
        if r <= cumulative_sum:
            selected_action_idx = legal_idx[i]
            break
    return int(selected_action_idx)


def render(black: int,
           white: int,
           player: int,
           highlight_legal: bool = True):
    """
    Bitboard 기반 시각화
    black: 흑 돌 bitboard
    white: 백 돌 bitboard
    player: 1 (black) or -1 (white)
    """

    if player == -1:
        black, white = white, black

    plt.figure(figsize=(5, 5))
    plt.imshow(np.zeros((BOARD_SIZE, BOARD_SIZE)),
               cmap="Greens", vmin=0, vmax=1)

    # grid
    for x in range(BOARD_SIZE + 1):
        plt.axhline(x - 0.5, color='k', linewidth=1)
        plt.axvline(x - 0.5, color='k', linewidth=1)

    # 돌 표시
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            idx = r * BOARD_SIZE + c
            bit = np.uint64(1) << np.uint64(idx)

            if black & bit:
                plt.plot(c, r, 'o', markersize=30, color='black')
            elif white & bit:
                plt.plot(
                    c, r, 'o',
                    markersize=30,
                    markeredgecolor='black',
                    markerfacecolor='white'
                )

    # legal move 표시
    if highlight_legal:
        if player == 1:
            own, opp = black, white
        else:
            own, opp = white, black

        legal_bits = get_legal_board(own, opp)

        legal = legal_bits
        while legal:
            # Find the index of the least significant bit (LSB)
            idx_found = -1
            for i in range(64):
                if (legal >> np.uint64(i)) & np.uint64(1):
                    idx_found = i
                    break
            if idx_found == -1:  # Should not happen if legal > 0
                break

            r = idx_found // BOARD_SIZE
            c = idx_found % BOARD_SIZE
            plt.plot(c, r, 'o', markersize=10, color='red', alpha=0.6)
            legal ^= (np.uint64(1) << np.uint64(idx_found))  # Clear the found bit

    plt.gca().invert_yaxis()
    plt.xticks(range(BOARD_SIZE))
    plt.yticks(range(BOARD_SIZE))
    if highlight_legal:
        plt.title(
            f"Current player: {'Black(●)' if player == 1 else 'White(○)'}"
        )
    plt.show()


class SectionTimer:
    def __init__(self, title=None):
        self.t = defaultdict(float)
        self.n = defaultdict(int)
        self.title = title

    def reset(self, title=None):
        self.t.clear()
        self.n.clear()
        self.title = title

    def add(self, key, dt):
        self.t[key] += time.perf_counter() - dt
        self.n[key] += 1

    def report(self):
        if self.title:
            print(f"======== {str(self.title)} ========")
        total = sum(self.t.values())
        for k in sorted(self.t, key=lambda x: -self.t[x]):
            avg = self.t[k] / max(1, self.n[k])
            print(f"{k:20s}: {self.t[k]:8.3f}s | avg {avg * 1e6:7.1f} µs")
        print(f"{'TOTAL':20s}: {total:8.3f}s")


timer = SectionTimer()


@contextmanager
def timed(timer: None | SectionTimer, label: str):
    if timer is None:
        yield
        return

    t0 = time.perf_counter()
    try:
        yield
    finally:
        timer.add(label, t0)
