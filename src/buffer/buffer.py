from src.mcts.mcts import *


class ReplayBuffer:
    def __init__(self, max_size=10_000):
        self.max_size = max_size
        self.ptr = 0

        self.own_buffer = np.zeros(max_size, dtype=np.uint64)
        self.opp_buffer = np.zeros(max_size, dtype=np.uint64)
        self.pi_buffer = np.zeros((max_size, ACTION_SIZE), dtype=np.float32)
        self.z_buffer = np.zeros(max_size, dtype=np.float32)

    def __len__(self):
        return min(self.max_size, self.ptr)

    def return_state(self):
        return (
            self.max_size, self.ptr,
            self.own_buffer, self.opp_buffer,
            self.pi_buffer, self.z_buffer
        )

    @classmethod
    def load_state(cls, state):
        tracker = cls(state[0])
        tracker.ptr = state[1]
        tracker.own_buffer = state[2]
        tracker.opp_buffer = state[3]
        tracker.pi_buffer = state[4]
        tracker.z_buffer = state[5]
        return tracker

    def add(self, own: int, opp: int, pi: np.ndarray, z: float):
        """
        Add (own, opp, pi, z) to buffer
        """
        idx = self.ptr % self.max_size

        self.own_buffer[idx] = own
        self.opp_buffer[idx] = opp
        self.pi_buffer[idx] = pi
        self.z_buffer[idx] = z

        self.ptr += 1

    def sample(self, batch_size) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Sample (states, pis, zs) from buffer
          - states: (B, 2, 8, 8)
          - pis: (B, 65)
          - zs: (B, )
        """
        size = len(self)
        if size == 0:
            raise RuntimeError("Replay buffer is empty")

        idx = np.random.randint(size, size=batch_size)

        owns = self.own_buffer[idx]
        opps = self.opp_buffer[idx]

        states_numpy = np.array(
            [bitboard_to_input(x, y) for (x, y) in zip(owns, opps)],
            dtype=np.float32
        )

        states = torch.from_numpy(states_numpy).to(DEVICE)
        pis = torch.from_numpy(self.pi_buffer[idx]).to(DEVICE)
        zs = torch.from_numpy(self.z_buffer[idx]).to(DEVICE)

        return states, pis, zs
