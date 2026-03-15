from src.mcts.mcts import *
from collections import deque


def generate_self_play(model: OthelloResNet, max_moves=128, timer=None):
    """
    Generate self_play history

    Returns
        - results: [(own, opp, pi, z, winner), ...]
        - winner
    """
    history: list[Tuple[int, int, np.ndarray, int]] = []
    results: list[Tuple[int, int, np.ndarray, float, int]] = []

    pass_count = 0
    own, opp = init_board()
    player = 1
    last_action = None
    mcts = MCTS(model)

    for move in range(max_moves):
        n_sim = 800 if move < 10 else 400 if move < 30 else 200
        temperature = 1.0 if move < 10 else 0.1 if move < 30 else 0

        with timed(timer, 'search'):
            pi = mcts.search(own, opp, n_sim=n_sim, last_action=last_action)

        history.append((own, opp, pi, player))

        with timed(timer, 'select_a_from_pi'):
            action = select_action_from_pi(pi, temperature)

        last_action = action

        with timed(timer, 'apply_move'):
            own, opp = apply_move_bitboard(own, opp, action)
            own, opp = opp, own
            player = -player

        pass_count = pass_count + 1 if action == PASS_ACTION else 0

        if pass_count == 2:
            break

    diff = popcount(own) - popcount(opp)
    winner = player if diff > 0 else -player if diff < 0 else 0

    with timed(timer, 'history'):
        for _own, _opp, pi, p in history:
            z = 0.0 if winner == 0 else (1.0 if winner == p else -1.0)
            results.append((_own, _opp, pi, z, p))

    return results, winner


def generate_game(policy_by_player: dict,
                  n_sim=50) -> int:
    """
    Duel with two models.

    policy_by_player[player] -> dict with model
      - Set random if None

    Returns
      - Winner
    """

    mcts_by_player = {
        p: None if m is None else MCTS(m, n_sim=n_sim, add_noise=False)
        for p, m in policy_by_player.items()
    }

    own, opp = init_board()
    player = 1
    pass_count = 0

    while True:
        mcts = mcts_by_player[player]
        if mcts is None:
            action = get_random_action(own, opp)

        else:
            pi = mcts.search(own, opp)
            action = select_action_from_pi(pi, 0)

        own, opp = apply_move_bitboard(own, opp, action)
        own, opp = opp, own
        player = -player

        pass_count = pass_count + 1 if action == PASS_ACTION else 0

        if pass_count == 2:
            break

    diff = popcount(own) - popcount(opp)
    winner = player if diff > 0 else -player if diff < 0 else 0
    return winner


class EloAgent:
    def __init__(self, init_elo=1500, K=16, window=4, plateau_delta=20):
        self.elos = defaultdict(float)  # model_id -> elo
        self.history = deque(maxlen=window)  # elos
        self.window = window
        self.K = K
        self.plateau_delta = plateau_delta
        self.init_elo = init_elo

    def state_dict(self):
        return {
            "elos": self.elos,
            "history": list(self.history),
            "K": self.K,
            "plateau_delta": self.plateau_delta,
            "window": self.history.maxlen,
            "init_elo": self.init_elo,
        }

    @classmethod
    def load_state_dict(cls, state):
        tracker = cls(
            init_elo=state["init_elo"],
            K=state["K"],
            window=state["window"],
            plateau_delta=state["plateau_delta"],
        )
        tracker.elos = state["elos"]
        tracker.history = deque(
            state["history"],
            maxlen=state["window"],
        )
        return tracker

    def expected(self, a_id: str, b_id: str) -> float:
        self.ensure(a_id, b_id)
        Ra, Rb = self.elos[a_id], self.elos[b_id]
        return 1 / (1 + 10 ** ((Rb - Ra) / 400))

    def record_iteration_delta(self, delta):
        self.history.append(delta)

    def is_plateau(self, win_rate=None):
        if len(self.history) < self.window:
            return False

        elo_flat = np.mean(np.abs(self.history)) < self.plateau_delta
        return elo_flat

    def ensure(self, *ids: str):
        for x in ids:
            if x not in self.elos:
                self.elos[x] = self.init_elo

    def update_game(self, id_a: str, id_b: str, result_a: float) -> float:
        """
        Record ids and results
          - id_a: player 1
          - id_b: player 2
          - result from a's perspective : 1.0 (win), 0.0 (lose), 0.5 (draw)
        Returns
          - delta
        """
        Ea = self.expected(id_a, id_b)
        delta = self.K * (result_a - Ea)

        if id_a != 'random':
            self.elos[id_a] += delta

        if id_b != 'random':
            self.elos[id_b] -= delta

        return delta


def duel(model_a, model_b,
         id_a='old', id_b='new',
         elo_agent: Optional[EloAgent] = None,
         n_games: int = 20, n_sim: int = 50,
         timer: Optional[SectionTimer] = None):
    stats = defaultdict(float)
    total_elo_delta_new = 0

    if elo_agent is None:
        elo_agent = EloAgent()

    if model_a is None:
        id_a = 'random'

    if model_b is None:
        id_b = 'random'

    for i in range(n_games):
        if i % 2 == 0:
            winner = generate_game({1: model_a, -1: model_b}, n_sim=n_sim)
            color_a = 1

        else:
            winner = generate_game({1: model_b, -1: model_a}, n_sim=n_sim)
            color_a = -1

        result_a = 1.0 if winner == color_a else 0.5 if winner == 0 else 0.0
        stats['history_a'] += result_a
        delta = elo_agent.update_game(id_a, id_b, result_a)
        total_elo_delta_new += delta

    stats['win_rate_a'] = stats['history_a'] / n_games
    stats['win_rate_b'] = 1 - stats['win_rate_a']

    elo_agent.record_iteration_delta(total_elo_delta_new)

    stats["elo_delta_new"] = total_elo_delta_new
    stats[id_a] = elo_agent.elos[id_a]
    stats[id_b] = elo_agent.elos[id_b]
    stats["plateau"] = float(elo_agent.is_plateau())

    return stats


if __name__ == "__main__":
    with timed(timer, 'duel'):
        duel(default_model, default_model)

    timer.report()

'''
def generate_duel_play(model_a: OthelloResNet,
                       model_b: OthelloResNet,
                       n_sim=50,
                       max_moves=128,
                       timer=None) -> int:
    """
    Duel with Two Models.

    Returns
      - winner
    """

    mcts_a = MCTS(model=model_a, add_noise=False, n_sim=n_sim)
    mcts_b = MCTS(model=model_b, add_noise=False, n_sim=n_sim)

    own, opp = init_board()
    player = 1
    pass_count = 0

    for move in range(max_moves):
        mcts = mcts_a if player == 1 else mcts_b
        pi = mcts.search(own, opp, timer=timer)
        action = select_action_from_pi(pi, 0)

        own, opp = apply_move_bitboard(own, opp, action)
        own, opp = opp, own
        player = -player

        pass_count = pass_count + 1 if action == PASS_ACTION else 0

        if pass_count == 2:
            break

    diff = popcount(own) - popcount(opp)
    winner = player if diff > 0 else -player if diff < 0 else 0
    return winner


def generate_random_play(model: OthelloResNet,
                         model_player: int = 1,
                         n_sim=50,
                         max_moves=128,
                         timer=None) -> int:
    assert model_player in (1, -1)

    own, opp = init_board()
    pass_count = 0
    mcts = MCTS(model, n_sim=n_sim, add_noise=False)

    if model_player == -1:
        legal = get_legal_board(own, opp)
        action = np.random.choice(legal)
        own, opp = apply_move_bitboard(own, opp, action)
        own, opp = opp, own

    while True:
        pi = mcts.search(own, opp)
        action = select_action_from_pi(pi, 0)
        own, opp = apply_move_bitboard(own, opp, action)
        own, opp = opp, own

        pass_count = pass_count + 1 if action == PASS_ACTION else 0
        if pass_count == 2:
            break

        action = np.random.choice(get_legal_board(own, opp))
        own, opp = apply_move_bitboard(own, opp, action)
        own, opp = opp, own
'''
