import torch
import torch.nn.functional as F
from torch import nn


# ===========================================================================
# Centralised Critic (CTDE -- centralised training, decentralised execution)
# ===========================================================================
# The critic sees the JOINT state of all agents and outputs a single value.
# Input convention: z, h are [B, N, latent] / [B, N, hidden] (Batch, Agent, F).
class Critic(nn.Module):
    def __init__(self, n_agents: int = 2, latent_dim: int = 32,
                 hidden_dim: int = 128, hidden_state: int = 256):
        super().__init__()
        self.n_agents = n_agents
        self.input_dim = n_agents * (latent_dim + hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, 1)
        )

    def forward(self, z, h, alive_mask=None):
        """z, h: [B, N, F]. alive_mask: optional [B, N] (1=alive).

        AUDIT FIX (D, "ghost agents"): when an agent is dead we must NOT feed
        its stale latent into the centralised critic, or the value function
        reads ghost agents. We zero out dead agents' slots so a dead agent
        contributes a constant zero instead of leaked state.
        """
        B, N, _ = z.shape
        if alive_mask is not None:
            m = alive_mask.unsqueeze(-1)          # [B, N, 1]
            z = z * m
            h = h * m
        z, h = z.reshape(B, -1), h.reshape(B, -1)  # [B, N*latent], [B, N*hidden]
        return self.net(torch.cat([z, h], dim=-1))  # [B, 1]


# ===========================================================================
# Communication modules. All take z, h of shape [B, N, F] and return a
# per-agent message/context of shape [B, N, message_dim].
# ===========================================================================
class NoCommunication(nn.Module):
    """No inter-agent communication: emits a zero context.

    AUDIT FIX (A4): the original stored a non-trainable `torch.zeros` weight
    matrix that (a) was never an nn.Parameter, (b) was never added to any
    optimizer, and (c) expressed "no communication" in a confusing way. We
    simply return a correctly-shaped zero tensor on the right device/dtype.
    """
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 32,
                 message_dim: int = 64):
        super().__init__()
        self.message_dim = message_dim

    def forward(self, z, h):
        return z.new_zeros(*z.shape[:-1], self.message_dim)


class BroadCastCommunication(nn.Module):
    """Each agent broadcasts an MLP-encoded message of its own state."""
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 32,
                 hidden_layer_dim: int = 86, message_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, message_dim)
        )

    def forward(self, z, h):
        message = torch.cat((z, h), dim=-1)
        return self.net(message)


class CommunicationHead(nn.Module):
    """Attention-based communication across the agent dimension."""
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 32,
                 QK_dim: int = 64, V_dim: int = 64):
        super().__init__()
        self.Q = nn.Linear(latent_dim + hidden_dim, QK_dim)
        self.K = nn.Linear(latent_dim + hidden_dim, QK_dim)
        self.V = nn.Linear(latent_dim + hidden_dim, V_dim)
        self.latent_dim = latent_dim + hidden_dim
        self.QK_dim = QK_dim
        self.V_dim = V_dim

    def forward(self, z, h, return_attn=False):
        # messages: [B, N, latent+hidden]; attention is over the agent dim N.
        messages = torch.cat((z, h), dim=-1)
        queries = self.Q(messages)                                   # [B, N, QK]
        keys = torch.transpose(self.K(messages), dim0=-2, dim1=-1)   # [B, QK, N]
        values = self.V(messages)                                    # [B, N, V]
        # AUDIT FIX (A5): the original used floor-division `//` for the scale,
        # which truncates the attention logits to integers BEFORE softmax,
        # destroying all sub-integer attention structure. Use true division.
        scores = (queries @ keys) / (self.QK_dim ** 0.5)             # [B, N, N]
        attentions = torch.softmax(scores, dim=-1)
        context = attentions @ values                                # [B, N, V]
        if return_attn:
            return context, attentions
        return context