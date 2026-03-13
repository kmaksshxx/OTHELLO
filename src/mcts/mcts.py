from src.environment import *
from src.models.models import OthelloResNet
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]
config_path = ROOT / 'configs' / 'config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

MCTS_SIMS = config['MCTS']['N_SIMS']
MCTS_BATCH = config['MCTS']['BATCH_SIZE']
MAX_DEPTH = config['MAX_DEPTH']
MAX_NODE = config['MCTS']['MAX_NODE']
default_model = OthelloResNet()


@nb.njit
def select_ucb(
        priors: np.ndarray, N: np.ndarray, W, legal_mask, c_puct, sum_N):
    best_a = -1
    best_score = -1e18
    sqrt_total = (sum_N + 1e-8) ** 0.5

    for a in range(priors.shape[0]):
        if legal_mask[a] == 0:
            continue

        Q = W[a] / N[a] if N[a] > 0 else 0.0
        U = c_puct * priors[a] * sqrt_total / (1.0 + N[a])
        score = Q + U

        if score > best_score:
            best_score = score
            best_a = a

    return best_a


@nb.njit
def get_or_create_child(
        nid, action, children, parent,
        node_count_ref, max_nodes, incoming_action):
    child = children[nid, action]

    if child != -1:
        return child

    child = node_count_ref[0]

    if child >= max_nodes:
        return -1

    node_count_ref[0] += 1

    children[nid, action] = child
    parent[child] = nid
    incoming_action[child] = action

    return child


@nb.njit
def backup_path(
        parent, incoming_action, path, path_len, N, W, sum_N, leaf_value):
    v = leaf_value
    for i in range(path_len - 1, 0, -1):
        cur = path[i]
        p = parent[cur]
        if p == -1:
            break
        a = incoming_action[cur]
        N[p, a] += 1
        sum_N[p] += 1
        W[p, a] += v
        v = -v


@nb.njit
def backup_path_batch(
        parent, incoming_action, paths, path_lens,
        N, W, sum_N, leaf_values):
    B = path_lens.shape[0]

    for b in range(B):
        p_len = path_lens[b]
        v = leaf_values[b]

        for i in range(p_len - 1, 0, -1):
            cur = paths[b, i]
            p = parent[cur]

            if p == -1:
                break

            a = incoming_action[cur]

            N[p, a] += 1
            sum_N[p] += 1
            W[p, a] += v

            v = -v


# @nb.njit(nb.int64(nb.uint64))
def popcount(x: int):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


class MCTS:
    def __init__(self, model, c_puct=1.5, n_sim=MCTS_SIMS,
                 batch_eval=MCTS_BATCH, dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 max_nodes=MAX_NODE, device=DEVICE, add_noise=True):
        self.model = model
        self.c_puct = c_puct
        self.n_sim = n_sim
        self.batch_eval = batch_eval
        self.d_a = dirichlet_alpha
        self.d_e = dirichlet_epsilon
        self.add_noise = add_noise
        self.max_nodes = max_nodes
        self.device = device

        self.root_nid = -1
        self.root_own = None
        self.root_opp = None

        # evaluation batch
        self.eval_pointer = np.array([0], np.int32)
        self.eval_nids = np.empty(batch_eval, np.int32)
        self.eval_paths = np.empty((batch_eval, MAX_DEPTH), np.int32)
        self.eval_path_lens = np.empty(batch_eval, np.int32)
        self.eval_states = np.empty((batch_eval, 2, 8, 8), np.float32)

        # tree
        self.parent = np.full(max_nodes, -1, np.int32)
        self.children = np.full((max_nodes, ACTION_SIZE), -1, np.int32)
        self.incoming_action = np.full(max_nodes, -1, np.int32)

        self.priors = np.zeros((max_nodes, ACTION_SIZE), np.float32)
        self.N = np.zeros((max_nodes, ACTION_SIZE), np.int32)
        self.W = np.zeros((max_nodes, ACTION_SIZE), np.float32)
        self.sum_N = np.zeros(max_nodes, np.int32)

        self.expanded = np.zeros(max_nodes, np.int8)
        self.node_count_ref = np.array([0], np.int32)

    def reset_pool(self):
        self.eval_pointer[0] = 0

        self.parent.fill(-1)
        self.children.fill(-1)
        self.incoming_action.fill(-1)

        self.priors.fill(0)
        self.N.fill(0)
        self.W.fill(0)
        self.sum_N.fill(0)

        self.expanded.fill(0)
        self.node_count_ref[0] = 0

    def _add_dirichlet_noise(self, nid, own, opp):
        """
        Apply Dirichlet noise to the Root Node (legal move only)
        """
        legal_bb = get_legal_board(own, opp)

        if legal_bb == 0:
            return

        moves = bitboard_to_array(legal_bb)
        k = len(moves)

        noise = np.random.dirichlet([self.d_a] * k).astype(np.float32)

        for i, m in enumerate(moves):
            self.priors[nid, m] = (
                    (1 - self.d_e) * self.priors[nid, m]
                    + self.d_e * noise[i]
            )

    def alloc_node(self) -> int:
        """Allocate Node"""
        nid = self.node_count_ref[0]
        if nid >= self.max_nodes:
            self.reset_pool()
            nid = 0
        self.node_count_ref[0] += 1
        return nid

    def ensure_root(self, own, opp, last_action: Optional[int] = None):
        # Try to reuse the tree if possible
        if self.root_nid != -1 and last_action is not None:
            child_nid = self.children[self.root_nid, last_action]
            if child_nid != -1 and self.expanded[child_nid] == 1:
                self.root_nid = child_nid
                self.root_own = own
                self.root_opp = opp
                if self.add_noise:
                    self._add_dirichlet_noise(self.root_nid, own, opp)
                return

        # If tree reuse failed or not attempted, reset and initialize a new root
        self.reset_pool()
        self.root_nid = self.alloc_node()
        self.root_own = own
        self.root_opp = opp

        inp = bitboard_to_input(own, opp)
        with torch.no_grad():
            p, v = self.model(
                torch.from_numpy(inp).unsqueeze(0).to(self.device)
            )

        self.priors[self.root_nid] = p.cpu().numpy().squeeze(0)
        self.expanded[self.root_nid] = 1

        if self.add_noise:
            self._add_dirichlet_noise(self.root_nid, own, opp)

        backup_path(
            self.parent, self.incoming_action,
            np.array([self.root_nid], np.int32),
            1, self.N, self.W, self.sum_N,
            v.cpu().numpy().item()
        )

    def search(self, own, opp, n_sim=None, last_action: Optional[int] = None,
               timer=None):
        if n_sim is None:
            n_sim = self.n_sim

        total_depth = 0

        with timed(timer, 'ensure_root'):
            self.ensure_root(own, opp, last_action)

        for sim in range(n_sim):
            with timed(timer, 'run_one_simulation'):
                depth = self.run_one_simulation(own, opp, timer)
                total_depth += depth

            if self.eval_pointer[0] >= self.batch_eval:
                with timed(timer, 'flush_eval_queue'):
                    self.flush_eval_queue()

            if timer and sim % 100 == 0 and sim > 0:
                print(
                    f"[MCTS] sim={sim:4d} | "
                    f"avg_depth={total_depth / (sim + 1):.2f} | "
                    f"expanded_nodes = {int((self.expanded == 1).sum())}"
                )

        with timed(timer, 'final_eval_queue'):
            self.flush_eval_queue()

        legal_bb = get_legal_board(own, opp)
        pi = np.zeros(ACTION_SIZE, np.float32)

        if legal_bb == 0:
            pi[PASS_ACTION] = 1.0
            return pi

        else:
            counts = self.N[self.root_nid].astype(np.float32)
            moves = bitboard_to_array(legal_bb)
            total = 0.0
            for m in moves:
                pi[m] = counts[m]
                total += pi[m]
            pi /= total

        return pi

    # ---------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------

    def run_one_simulation(self, root_own, root_opp,
                           timer: Optional[SectionTimer] = None):
        nid = self.root_nid
        own, opp = root_own, root_opp

        path = np.empty(MAX_DEPTH, np.int32)
        path_len = 0
        path[path_len] = nid
        path_len += 1

        depth = 0

        while True:
            with timed(timer, 'terminal_check'):
                legal = get_legal_board(own, opp)
                opp_legal = get_legal_board(opp, own)

            if legal == 0 and opp_legal == 0:
                # terminal
                with timed(timer, 'popcount'):
                    own_count = popcount(own)
                    opp_count = popcount(opp)
                v = 1.0 if own_count > opp_count else -1.0 if own_count < opp_count else 0.0

                with timed(timer, 'backup_path'):
                    backup_path(
                        self.parent, self.incoming_action,
                        path, path_len,
                        self.N, self.W, self.sum_N, v
                    )
                return depth

            # -------------------------
            # leaf
            # -------------------------
            if self.expanded[nid] == 0:
                p = self.eval_pointer[0]

                self.eval_nids[p] = nid
                self.eval_paths[p, :path_len] = path[:path_len]
                self.eval_path_lens[p] = path_len
                self.eval_states[p] = bitboard_to_input(own, opp)[0]

                self.eval_pointer[0] += 1
                self.expanded[nid] = -1
                return depth

            # -------------------------
            # selection
            # -------------------------
            if legal == 0:
                a = PASS_ACTION
            else:
                legal_mask = np.zeros(ACTION_SIZE, np.uint8)
                for m in bitboard_to_array(np.uint64(legal)):
                    legal_mask[m] = 1

                with timed(timer, 'select_ucb'):
                    a = select_ucb(
                        self.priors[nid],
                        self.N[nid],
                        self.W[nid],
                        legal_mask,
                        self.c_puct,
                        self.sum_N[nid]
                    )

            with timed(timer, 'get_or_create_child'):
                child = get_or_create_child(
                    nid, a,
                    self.children,
                    self.parent,
                    self.node_count_ref,
                    self.max_nodes,
                    self.incoming_action
                )

            if child == -1:
                with timed(timer, 'reset_pool'):
                    self.reset_pool()
                return depth

            # apply
            own, opp = apply_move_bitboard(own, opp, a)
            own, opp = opp, own  # player switch

            nid = child
            path[path_len] = nid
            path_len += 1
            depth += 1

            if depth >= MAX_DEPTH:
                return depth

    # ---------------------------------------------------------
    # Batch NN eval
    # ---------------------------------------------------------

    def flush_eval_queue(self):
        B = self.eval_pointer[0]
        if B == 0:
            return

        self.eval_pointer[0] = 0

        with torch.no_grad():
            p, v = self.model(
                torch.from_numpy(self.eval_states[:B]).to(self.device)
            )

        p = p.cpu().numpy()
        v = v.cpu().numpy().squeeze(-1)

        for i in range(B):
            nid = self.eval_nids[i]
            self.priors[nid] = p[i]
            self.expanded[nid] = 1

        backup_path_batch(
            self.parent,
            self.incoming_action,
            self.eval_paths[:B],
            self.eval_path_lens[:B],
            self.N,
            self.W,
            self.sum_N,
            v[:B]
        )

    def reset_tree(self):
        self.reset_pool()
        self.root_nid = -1


if __name__ == '__main__':
    mcts = MCTS(default_model, n_sim=MCTS_SIMS, batch_eval=MCTS_BATCH)
    timer = SectionTimer('MCTS')
    own, opp = init_board()
    mcts.search(own, opp, timer=timer)
    timer.report()

    # _mcts.reset_tree()
    #
    # pass_count = 0
    # own, opp = init_board
    # action = None
    # player = 1
    #
    # while True:
    #     pi = _mcts.search(own, opp, last_action_by_prev_player=action)
    #     action = select_action_from_pi(pi, temperature=0.01)
    #     own, opp = apply_move_bitboard(own, opp, action)
    #     own, opp = opp, own
    #     player = -player
    #
    #     pass_count = pass_count + 1 if action == PASS_ACTION else 0
    #
    #     if pass_count == 2:
    #         break
    #
    # render(own, opp, player)
