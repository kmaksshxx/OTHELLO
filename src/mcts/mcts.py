from src.environment import *
from src.models.models import OthelloResNet
from pathlib import Path
import threading
import yaml

ROOT = Path(__file__).resolve().parents[2]
config_path = ROOT / 'configs' / 'config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

MCTS_SIMS = config['MCTS']['N_SIMS']
BATCH_SIZE = config['BATCH_SIZE']
MAX_DEPTH = config['MAX_DEPTH']
MAX_NODE = config['MCTS']['MAX_NODE']
n_workers = config['MCTS']['N_WORKERS']

default_model = OthelloResNet()
default_model.to(DEVICE)


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


def popcount(x: int):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


class _MCTS:
    def __init__(self, model: OthelloResNet, c_puct=1.5, n_sim=MCTS_SIMS,
                 batch_eval=BATCH_SIZE, dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 max_nodes=MAX_NODE, device=DEVICE, add_noise=True):
        self.model = model
        self.model.to(DEVICE)

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


class MCTS:
    def __init__(
            self,
            model: OthelloResNet,
            c_puct: float = 1.5,
            n_sim: int = MCTS_SIMS,
            batch_eval: int = BATCH_SIZE,
            dirichlet_alpha: float = 0.3,
            dirichlet_epsilon: float = 0.25,
            max_nodes: int = MAX_NODE,
            device=DEVICE,
            add_noise: bool = True,
            n_workers: int = n_workers,
            virtual_loss: int = 3,
            timer: Optional[SectionTimer] = None
    ):
        self.model = model
        self.model.to(device)

        self.c_puct = c_puct
        self.n_sim = n_sim
        self.batch_eval = batch_eval
        self.d_a = dirichlet_alpha
        self.d_e = dirichlet_epsilon
        self.add_noise = add_noise
        self.max_nodes = max_nodes
        self.device = device
        self.n_workers = n_workers
        self.virtual_loss = virtual_loss

        self.root_nid: int = -1
        self.root_own = None
        self.root_opp = None

        # ── Tree arrays ──────────────────────────────────────────────────────
        self.parent = np.full(max_nodes, -1, np.int32)
        self.children = np.full((max_nodes, ACTION_SIZE), -1, np.int32)
        self.incoming_action = np.full(max_nodes, -1, np.int32)

        self.priors = np.zeros((max_nodes, ACTION_SIZE), np.float32)
        self.N = np.zeros((max_nodes, ACTION_SIZE), np.int32)
        self.W = np.zeros((max_nodes, ACTION_SIZE), np.float32)
        self.sum_N = np.zeros(max_nodes, np.int32)

        # expanded:  0 = fresh, -1 = queued for eval, 1 = expanded
        self.expanded = np.zeros(max_nodes, np.int8)
        self.node_count_ref = np.array([0], np.int32)

        # ── Concurrency primitives ────────────────────────────────────────────
        # _queue_cap: maximum leaves that can sit in the queue at once.
        # Sized to hold exactly one full batch per worker so no worker ever
        # has to spin-wait under normal operation.
        self._queue_cap = batch_eval * n_workers

        # Single Condition whose lock replaces _alloc_lock everywhere.
        # All queue state (ptr, arrays) and node-alloc state are protected by it.
        self._cv = threading.Condition(threading.Lock())

        # Leaf eval queue (protected by _cv)
        self._eval_nids = np.empty(self._queue_cap, np.int32)
        self._eval_paths = np.empty((self._queue_cap, MAX_DEPTH), np.int32)
        self._eval_path_lens = np.empty(self._queue_cap, np.int32)
        self._eval_states = np.empty((self._queue_cap, 2, 8, 8), np.float32)
        self._eval_ptr = 0  # guarded by _cv

        # Flag set by coordinator while a flush is in progress.
        # Workers wait on _cv when this is True.
        self._flushing = False

    # ─────────────────────────────────────────────────────────────────────────
    # Pool management
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_pool(self):
        self.parent.fill(-1)
        self.children.fill(-1)
        self.incoming_action.fill(-1)
        self.priors.fill(0)
        self.N.fill(0)
        self.W.fill(0)
        self.sum_N.fill(0)
        self.expanded.fill(0)
        self.node_count_ref[0] = 0
        self._eval_ptr = 0

    def _alloc_node(self) -> int:
        """Allocate a node ID.  Caller must hold _cv."""
        nid = self.node_count_ref[0]
        if nid >= self.max_nodes:
            # Pool exhausted mid-search: reset and start fresh.
            # This is rare if max_nodes is sized generously.
            self._reset_pool()
            nid = 0
        self.node_count_ref[0] += 1
        return nid

    # ─────────────────────────────────────────────────────────────────────────
    # Dirichlet noise
    # ─────────────────────────────────────────────────────────────────────────

    def _add_dirichlet_noise(self, nid, own, opp):
        legal_bb = get_legal_board(own, opp)
        if legal_bb == 0:
            return
        moves = bitboard_to_array(legal_bb)
        noise = np.random.dirichlet([self.d_a] * len(moves)).astype(np.float32)
        for i, m in enumerate(moves):
            self.priors[nid, m] = (
                    (1 - self.d_e) * self.priors[nid, m] + self.d_e * noise[i]
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Deep tree reuse  (subtree compaction)
    # ─────────────────────────────────────────────────────────────────────────

    def _compact_subtree(self, new_root_nid: int):
        """
        Copy the subtree rooted at new_root_nid into a fresh pool using a BFS
        remap, so node IDs stay dense and small.

        old_nid → new_nid mapping is built during BFS.
        After compaction, self.root_nid == 0 and all arrays are tightly packed.
        """
        max_n = self.max_nodes

        # Temporary storage for the compacted tree
        new_parent = np.full(max_n, -1, np.int32)
        new_children = np.full((max_n, ACTION_SIZE), -1, np.int32)
        new_incoming_action = np.full(max_n, -1, np.int32)
        new_priors = np.zeros((max_n, ACTION_SIZE), np.float32)
        new_N = np.zeros((max_n, ACTION_SIZE), np.int32)
        new_W = np.zeros((max_n, ACTION_SIZE), np.float32)
        new_sum_N = np.zeros(max_n, np.int32)
        new_expanded = np.zeros(max_n, np.int8)

        remap = {}  # old_nid → new_nid
        queue = [new_root_nid]
        counter = 0

        while queue:
            old = queue.pop(0)
            new = counter
            counter += 1
            remap[old] = new

            new_priors[new] = self.priors[old]
            new_N[new] = self.N[old]
            new_W[new] = self.W[old]
            new_sum_N[new] = self.sum_N[old]
            new_expanded[new] = self.expanded[old]
            new_incoming_action[new] = self.incoming_action[old]

            for a in range(ACTION_SIZE):
                child_old = self.children[old, a]
                if child_old != -1 and self.expanded[child_old] == 1:
                    queue.append(child_old)

        # Second pass: fix up child and parent pointers
        for old, new in remap.items():
            for a in range(ACTION_SIZE):
                child_old = self.children[old, a]
                if child_old in remap:
                    new_children[new, a] = remap[child_old]
                    new_parent[remap[child_old]] = new

        # Swap in
        self.parent = new_parent
        self.children = new_children
        self.incoming_action = new_incoming_action
        self.priors = new_priors
        self.N = new_N
        self.W = new_W
        self.sum_N = new_sum_N
        self.expanded = new_expanded
        self.node_count_ref[0] = counter

        # New root is always node 0 after compaction
        self.root_nid = 0
        # Root has no parent
        self.parent[0] = -1
        self.incoming_action[0] = -1

    def ensure_root(self, own, opp, last_action: Optional[int] = None):
        if self.root_nid != -1 and last_action is not None:
            child_nid = self.children[self.root_nid, last_action]
            if child_nid != -1 and self.expanded[child_nid] == 1:
                # ── Deep tree reuse: compact the surviving subtree ──────────
                self._compact_subtree(child_nid)
                self.root_own = own
                self.root_opp = opp
                if self.add_noise:
                    self._add_dirichlet_noise(self.root_nid, own, opp)
                return

        # Fresh start
        self._reset_pool()
        with self._cv:
            self.root_nid = self._alloc_node()
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
            np.array([self.root_nid], np.int32), 1,
            self.N, self.W, self.sum_N,
            v.cpu().numpy().item(),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Virtual loss helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_virtual_loss(self, path: np.ndarray, path_len: int):
        """
        Walk the path and subtract VIRTUAL_LOSS from W along each edge,
        and add VIRTUAL_LOSS to N / sum_N so UCB is repelled.
        This is done before the NN eval so concurrent threads see the penalty.
        """
        vl = self.virtual_loss
        for i in range(path_len - 1):
            nid = path[i]
            a = self.incoming_action[path[i + 1]]
            self.N[nid, a] += vl
            self.W[nid, a] -= vl
            self.sum_N[nid] += vl

    def _revert_virtual_loss(self, path: np.ndarray, path_len: int):
        """
        Undo virtual loss along the path.  Called inside flush_eval_queue
        immediately before the real backup so the net effect is correct.
        """
        vl = self.virtual_loss
        for i in range(path_len - 1):
            nid = path[i]
            a = self.incoming_action[path[i + 1]]
            self.N[nid, a] -= vl
            self.W[nid, a] += vl
            self.sum_N[nid] -= vl

    # ─────────────────────────────────────────────────────────────────────────
    # One simulation (runs in a worker thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_one_simulation(self, root_own, root_opp):
        nid = self.root_nid
        own, opp = root_own, root_opp

        path = np.empty(MAX_DEPTH, np.int32)
        path_len = 0
        path[path_len] = nid
        path_len += 1

        depth = 0

        while True:
            legal = get_legal_board(own, opp)
            opp_legal = get_legal_board(opp, own)

            # ── Terminal ──────────────────────────────────────────────────────
            if legal == 0 and opp_legal == 0:
                own_count = popcount(own)
                opp_count = popcount(opp)
                v = (
                    1.0 if own_count > opp_count
                    else -1.0 if own_count < opp_count
                    else 0.0)
                backup_path(
                    self.parent, self.incoming_action,
                    path, path_len,
                    self.N, self.W, self.sum_N, v,
                )
                return

            # ── Leaf (unexpanded) ─────────────────────────────────────────────
            if self.expanded[nid] == 0:
                with self._cv:
                    # Block while a flush is in progress or the queue is full.
                    # Both conditions are guarded by the same _cv lock so there
                    # is no window where _eval_ptr can overflow.
                    while self._flushing or self._eval_ptr >= self._queue_cap:
                        self._cv.wait()

                    # Double-check: another thread may have expanded this node
                    # while we were waiting on _cv.
                    if self.expanded[nid] == 0:
                        self.expanded[nid] = -1  # mark as queued

                        p = self._eval_ptr
                        self._eval_nids[p] = nid
                        self._eval_paths[p, :path_len] = path[:path_len]
                        self._eval_path_lens[p] = path_len
                        self._eval_states[p] = bitboard_to_input(own, opp)[0]
                        self._eval_ptr += 1

                        self._apply_virtual_loss(path, path_len)

                        # Wake coordinator if we've reached a full batch
                        if self._eval_ptr >= self.batch_eval:
                            self._cv.notify_all()

                        return
                    # else: node was expanded by another thread; fall through
                    # to selection without returning.

            # ── Selection ─────────────────────────────────────────────────────
            if legal == 0:
                a = PASS_ACTION
            else:
                legal_mask = np.zeros(ACTION_SIZE, np.uint8)
                for m in bitboard_to_array(np.uint64(legal)):
                    legal_mask[m] = 1

                a = select_ucb(
                    self.priors[nid],
                    self.N[nid],
                    self.W[nid],
                    legal_mask,
                    self.c_puct,
                    self.sum_N[nid],
                )

            # ── Expand child ──────────────────────────────────────────────────
            with self._cv:
                child = get_or_create_child(
                    nid, a,
                    self.children, self.parent,
                    self.node_count_ref, self.max_nodes,
                    self.incoming_action,
                )

            if child == -1:
                with self._cv:
                    self._reset_pool()
                return

            own, opp = apply_move_bitboard(own, opp, a)
            own, opp = opp, own

            nid = child
            path[path_len] = nid
            path_len += 1
            depth += 1

            if depth >= MAX_DEPTH:
                return

    # ─────────────────────────────────────────────────────────────────────────
    # Batch NN eval  (runs on the main thread / coordinator)
    # ─────────────────────────────────────────────────────────────────────────

    def _flush_eval_queue(self):
        with self._cv:
            B = self._eval_ptr
            if B == 0:
                return
            # Snapshot and clear the queue atomically, then raise flushing flag
            # so workers block while the GPU call is in progress.
            nids = self._eval_nids[:B].copy()
            paths = self._eval_paths[:B].copy()
            path_lens = self._eval_path_lens[:B].copy()
            states = self._eval_states[:B].copy()
            self._eval_ptr = 0
            self._flushing = True

        # GPU forward pass — outside _cv so workers can hold it if needed.
        with torch.no_grad():
            p_batch, v_batch = self.model(
                torch.from_numpy(states).to(self.device)
            )
        p_batch = p_batch.cpu().numpy()
        v_batch = v_batch.cpu().numpy().squeeze(-1)

        # Write results back and revert virtual loss under _cv.
        with self._cv:
            for i in range(B):
                self._revert_virtual_loss(paths[i], path_lens[i])
                self.priors[nids[i]] = p_batch[i]
                self.expanded[nids[i]] = 1

            backup_path_batch(
                self.parent, self.incoming_action,
                paths, path_lens,
                self.N, self.W, self.sum_N,
                v_batch,
            )

            # Clear flushing flag and wake all waiting workers.
            self._flushing = False
            self._cv.notify_all()

    # ─────────────────────────────────────────────────────────────────────────
    # Public search entry point
    # ─────────────────────────────────────────────────────────────────────────

    def search(
            self, own, opp,
            n_sim: Optional[int] = None,
            last_action: Optional[int] = None,
            timer: Optional[SectionTimer] = None,
    ) -> np.ndarray:
        if n_sim is None:
            n_sim = self.n_sim

        with timed(timer, 'ensure_root'):
            self.ensure_root(own, opp, last_action)

        with timed(timer, 'distribution'):
            # Distribute simulations across workers as evenly as possible.
            # Each worker runs its share sequentially; all workers run concurrently.
            sims_per_worker = [n_sim // self.n_workers] * self.n_workers
            for i in range(n_sim % self.n_workers):
                sims_per_worker[i] += 1

            # Shared counter: coordinator exits when it reaches zero.
            self._workers_remaining = sum(sims_per_worker)
            self._all_done = False

        with timed(timer, 'define function'):
            def worker(n: int):
                for _ in range(n):
                    self._run_one_simulation(own, opp)
                # Decrement shared counter and wake coordinator if last worker done.
                with self._cv:
                    self._workers_remaining -= n
                    if self._workers_remaining <= 0:
                        self._all_done = True
                        self._cv.notify_all()

            def coordinator():
                while True:
                    with self._cv:
                        # Wait until a full batch is ready OR all workers finished.
                        self._cv.wait_for(
                            lambda: self._eval_ptr >= self.batch_eval or self._all_done
                        )
                        done = self._all_done

                    self._flush_eval_queue()

                    if done:
                        # One final flush in case workers enqueued more leaves
                        # between the flag being set and us checking it.
                        self._flush_eval_queue()
                        break

        with timed(timer, 'thread'):
            threads = []
            for n in sims_per_worker:
                t = threading.Thread(target=worker, args=(n,), daemon=True)
                t.start()
                threads.append(t)

        with timed(timer, 'coordinator'):
            coordinator()

        with timed(timer, 'join'):
            for t in threads:
                t.join()

        with timed(timer, 'build'):
            # ── Build policy ──────────────────────────────────────────────────────
            legal_bb = get_legal_board(own, opp)
            pi = np.zeros(ACTION_SIZE, np.float32)

            if legal_bb == 0:
                pi[PASS_ACTION] = 1.0
            else:
                counts = self.N[self.root_nid].astype(np.float32)
                moves = bitboard_to_array(legal_bb)
                total = sum(counts[m] for m in moves)
                for m in moves:
                    pi[m] = counts[m] / total if total > 0 else 0.0

        return pi

    def reset_tree(self):
        self._reset_pool()
        self.root_nid = -1


if __name__ == '__main__':
    timer.reset('MCTS')
    own, opp = init_board()

    mcts = MCTS(default_model)
    pi = mcts.search(own, opp, timer=timer)
    timer.report()

    print(pi)
