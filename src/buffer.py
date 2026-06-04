from collections import deque
import random
import torch


# ===========================================================================
# Sequence replay buffer for recurrent / world-model training.
# ===========================================================================
# every episode is stored as a dict of PADDED tensors
# in the canonical [T, N, F] layout (Time, Agent, Feature) plus two masks:
#   seq_mask  [T]      : 1 where timestep t is a real (non-padded) step.
#   alive_mask[T, N]   : 1 where agent n is alive (participating) at step t.
#
# Sampling stacks `batch_size` episodes along a new leading batch dim, padding
# to the longest episode in the batch, producing the canonical [B, T, N, F]
# tensors + [B, T] and [B, T, N] masks consumed by the trainer.
class SequenceBufferReplay:
    def __init__(self, max_len: int = 5000):
        self.buffer = deque(maxlen=max_len)

    def __len__(self):
        return len(self.buffer)

    def append(self, episode: dict):
        """Store one episode.

        `episode` is a dict with keys:
            'obs'   : FloatTensor [T, N, obs_dim]
            'act'   : FloatTensor [T, N, action_dim]   (one-hot)
            'rew'   : FloatTensor [T, N]
            'done'  : FloatTensor [T, N]               (per-agent terminal)
            'seq_mask'   : FloatTensor [T]             (1 = real step)
            'alive_mask' : FloatTensor [T, N]          (1 = agent alive)
        """
        self.buffer.append(episode)

    def sample(self, batch_size: int):
        """Sample `batch_size` episodes, pad to the max T in the batch, stack.

        Returns a dict of tensors with a leading batch dim B:
            'obs'        : [B, T, N, obs_dim]
            'act'        : [B, T, N, action_dim]
            'rew'        : [B, T, N]
            'done'       : [B, T, N]
            'seq_mask'   : [B, T]
            'alive_mask' : [B, T, N]
        """
        episodes = random.choices(self.buffer, k=batch_size)

        max_T = max(ep["obs"].shape[0] for ep in episodes)
        N = episodes[0]["obs"].shape[1]

        def pad_time(x, target_T):
            T = x.shape[0]
            if T == target_T:
                return x
            pad_shape = (target_T - T, *x.shape[1:])
            pad = x.new_zeros(pad_shape)
            return torch.cat([x, pad], dim=0)

        batch = {k: [] for k in
                 ("obs", "act", "rew", "done", "seq_mask", "alive_mask")}
        for ep in episodes:
            for k in batch:
                batch[k].append(pad_time(ep[k], max_T))

        out = {k: torch.stack(v, dim=0) for k, v in batch.items()}
        assert out["obs"].shape[:2] == out["seq_mask"].shape, \
            (out["obs"].shape, out["seq_mask"].shape)
        return out
