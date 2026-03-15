from src.mcts.mcts import *


class ReplayBuffer:
    def __init__(self, max_size=100_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.own_buffer = np.zeros(max_size, dtype=np.uint64)
        self.opp_buffer = np.zeros(max_size, dtype=np.uint64)

        self.pi_buffer = np.zeros((max_size, ACTION_SIZE), dtype=np.float32)
        self.z_buffer = np.zeros(max_size, dtype=np.float32)

    def __len__(self):
        return self.size

    def add(self, own: int, opp: int, pi: np.ndarray, z: float):
        # assert own < 2 ** 64 and opp < 2 ** 64
        idx = self.ptr % self.max_size

        self.own_buffer[idx] = own
        self.opp_buffer[idx] = opp
        self.pi_buffer[idx] = pi
        self.z_buffer[idx] = z

        self.ptr += 1
        if self.size < self.max_size:
            self.size += 1

    def sample(self, batch_size, recent_frac=0.3, recent_prob=0.5) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        if self.size == 0:
            raise RuntimeError("Replay buffer is empty")

        recent_size = int(self.size * recent_frac)
        recent_start = (self.ptr - recent_size) % self.max_size

        idxs = []

        n_recent = int(batch_size * recent_prob)
        n_uniform = batch_size - n_recent

        # recent bucket sampling
        for _ in range(n_recent):
            offset = np.random.randint(recent_size)
            idxs.append((recent_start + offset) % self.max_size)

        # uniform sampling
        idxs.extend(np.random.randint(0, self.size, size=n_uniform))

        idxs = np.array(idxs, dtype=np.int64)

        owns = self.own_buffer[idxs]
        opps = self.opp_buffer[idxs]

        states_numpy = np.array(
            [bitboard_to_input(x, y) for (x, y) in zip(owns, opps)],
            dtype=np.float32
        )

        if DEVICE == 'cuda':
            states = torch.from_numpy(states_numpy).pin_memory()
            states = states.to(DEVICE, non_blocking=True)

            pis = torch.from_numpy(self.pi_buffer[idxs]).pin_memory()
            pis = pis.to(DEVICE, non_blocking=True)

            zs = torch.from_numpy(self.z_buffer[idxs]).pin_memory()
            zs = zs.to(DEVICE, non_blocking=True)
        else:
            states = torch.from_numpy(states_numpy)
            pis = torch.from_numpy(self.pi_buffer[idxs]).to(DEVICE)
            zs = torch.from_numpy(self.z_buffer[idxs]).to(DEVICE)

        return states, pis, zs


# def soft_reset(self, keep_frac=0.4):
#     """
#     Keep only the most recent keep_frac portion of the buffer.
#     Old data are discarded. Pointer and size are updated accordingly.
#     """

#     if self.size == 0:
#         return

#     keep_size = max(int(self.size * keep_frac), 1)

#     # recent data start index (ring buffer aware)
#     start = (self.ptr - keep_size) % self.max_size

#     if start + keep_size <= self.max_size:
#         # contiguous case
#         states = self.state_buffer[start:start + keep_size].copy()
#         pis = self.pi_buffer[start:start + keep_size].copy()
#         zs = self.z_buffer[start:start + keep_size].copy()
#     else:
#         # wrapped case
#         first_len = self.max_size - start
#         second_len = keep_size - first_len

#         states = np.concatenate([
#             self.state_buffer[start:].copy(),
#             self.state_buffer[:second_len].copy()
#         ], axis=0)

#         pis = np.concatenate([
#             self.pi_buffer[start:].copy(),
#             self.pi_buffer[:second_len].copy()
#         ], axis=0)

#         zs = np.concatenate([
#             self.z_buffer[start:].copy(),
#             self.z_buffer[:second_len].copy()
#         ], axis=0)

#     # overwrite buffer front
#     self.state_buffer[:keep_size] = states
#     self.pi_buffer[:keep_size] = pis
#     self.z_buffer[:keep_size] = zs

#     self.size = keep_size
#     self.ptr = keep_size

