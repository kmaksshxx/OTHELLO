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
    own, opp = init_board
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

    diff = popcount(own) - popcount(opp)
    winner = player if diff > 0 else -player if diff < 0 else 0

    with timed(timer, 'history'):
        for _own, _opp, pi, p in history:
            z = 0.0 if winner == 0 else (1.0 if winner == p else -1.0)
            results.append((_own, _opp, pi, z, p))

    return results, winner


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

    own, opp = init_board
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

    own, opp = init_board
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
def generate_game(old_model: Optional[OthelloResNet],
                  new_model: Optional[OthelloResNet],
                  max_moves=128,
                  n_sim=50,
                  duel=False,
                  timer=None):
    """
    Duel with two models. Set random if model is None

    Returns
        - results: [(s, pi, z, winner), ...]
        - winner
    """
    own, opp = init_board
    player = 1
    history: list[Tuple[int, int, np.ndarray, int]] = []
    results: list[Tuple[int, int, np.ndarray, float, int]] = []
    pass_count = 0
    old_mcts = None if old_model is None else MCTS(old_model, n_sim=n_sim, add_noise=False)
    new_mcts = None if new_model is None else MCTS(new_model, n_sim=n_sim, add_noise=False)

    for move in range(max_moves):
        policy = policy_by_player[player]
        mcts = policy.get("mcts", None)

        if mcts is not None:
            # set n_sim and temperature
            mcts.n_sim = n_sim if duel else 800 if move < 10 else 400 if move < 30 else 200
            temperature = 0 if duel else 1.0 if move < 10 else 0.1 if move < 30 else 0

            with timed(timer, 'search'):
                pi = mcts.search(own, opp)

            with timed(timer, 'select_a_from_pi'):
                action = select_action_from_pi(pi, temperature)

            if policy.get('record', True) and not duel:
                # Use bitboard representation for history
                history.append((own, opp, pi, player))

        # Random Policy
        else:
            with timed(timer, 'legal'):
                legal = get_legal_board(own, opp)

            if legal == 0:
                action = PASS_ACTION
            else:
                # bitboard_to_array returns np.int64 array, np.random.choice handles it
                moves = bitboard_to_array(legal)
                action = np.random.choice(moves)

        last_action_by_prev_player = action

        with timed(timer, 'apply_move'):
            own, opp = apply_move_bitboard(own, opp, action)
            own, opp = opp, own
            player = -player

        pass_count = pass_count + 1 if action == PASS_ACTION else 0

        if pass_count == 2:
            break

    own_count = popcount(own)
    opp_count = popcount(opp)
    diff = own_count - opp_count
    winner = player if diff > 0 else -player if diff < 0 else 0

    with timed(timer, 'history'):
        for _own, _opp, pi, p in history:
            z = 0.0 if winner == 0 else (1.0 if winner == p else -1.0)
            # When adding to replay buffer, use bitboard directly
            # Convert to board array only if needed later, or adapt replay buffer to store bitboards
            results.append((_own, _opp, pi, z, p))  # Store bitboards directly

    return results, winner

'''


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

    def update_game(self, a_id: str, b_id: str, result_a: int | float, freeze_b=False) -> float:
        """
        Record ids and results
          - a_id: player 1
          - b_id: player 2
          - result from a's perspective : 0.0 (draw), 1.0 (win), -1.0 (lose)
        Returns
          - delta
        """
        Ea = self.expected(a_id, b_id)
        delta = self.K * (float(result_a) - Ea)
        self.elos[a_id] += delta

        if not freeze_b:
            self.elos[b_id] -= delta

        return delta


def duel(old_model, new_model,
         old_id='old', new_id='new',
         elo_agent: Optional[EloAgent] = None,
         n_games: int = 20, n_sim: int = 50,
         timer: Optional[SectionTimer] = None):
    stats = defaultdict(float)
    total_elo_delta_new = 0

    if elo_agent is None:
        elo_agent = EloAgent()

    if old_model is None:
        old_id = 'random'

    mcts_old = None if old_model is None else MCTS(old_model, add_noise=False, n_sim=n_sim)
    mcts_new = MCTS(new_model, add_noise=False, n_sim=n_sim)

    for i in range(n_games):
        if mcts_old:
            mcts_old.reset_tree()
        mcts_new.reset_tree()

        if i % 2 == 0:
            players = {1: mcts_old, -1: mcts_new}
            old_color = 1
        else:
            players = {1: mcts_new, -1: mcts_old}
            old_color = -1

        _, winner = generate_game({
            p: {'mcts': m, 'record': False}
            for p, m in players.items()
        }, duel=True, timer=timer)

        if winner == 0:
            stats["draw"] += 1.0
            Sb_new = 0.5
        elif winner == old_color:
            stats["old_win"] += 1.0
            Sb_new = 0.0
        else:
            stats["new_win"] += 1.0
            Sb_new = 1.0

        delta = elo_agent.update_game(new_id, old_id, Sb_new, (old_model is None))
        total_elo_delta_new += delta

    stats["win_rate_old"] = (
        stats["old_win"] + 0.5 * stats["draw"]
        ) / n_games
    stats["win_rate_new"] = 1 - stats["win_rate_old"]

    elo_agent.record_iteration_delta(total_elo_delta_new)

    stats["elo_delta_new"] = total_elo_delta_new
    stats["elo_new"] = elo_agent.elos[new_id]
    stats["elo_old"] = elo_agent.elos[old_id]
    stats["plateau"] = float(elo_agent.is_plateau())

    return stats
