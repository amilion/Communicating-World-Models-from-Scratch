import torch
import torch.nn.functional as F
from torch import nn


# ===========================================================================
# RSSM (Recurrent State-Space Model)
# ===========================================================================
# This is the latent dynamics core (PlaNet/Dreamer style):
#   h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})        (deterministic recurrent state)
#   prior     p(z_t | h_t)                         (dynamics prediction)
#   posterior q(z_t | h_t, embed(o_t))             (observation-conditioned)
class RSSM(nn.Module):
    def __init__(self,
                 latent_dim: int = 32,
                 action_dim: int = 5,
                 hidden_dim: int = 128,
                 obs_embed: int = 256):
        super().__init__()
        self.num_gru_layers = 1
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.gru_unit = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + obs_embed, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)
        )

        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)
        )

    def initial_states(self, batch_size, device=None):
        """Return zero (latent_state, hidden_state) for a fresh sequence.
        """
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        return z, h

    def split_params(self, params):
        mu, sigma = torch.chunk(params, 2, dim=-1)
        sigma = F.softplus(sigma) + 1e-4
        return mu, sigma

    def latent_representation(self, mu, sigma):
        return mu + torch.randn_like(sigma) * sigma

    def _step_h(self, z_prev, h_prev, action):
        """One deterministic GRU step.
        """
        action = action.reshape(action.shape[0], -1)
        gru_input = torch.cat([z_prev, action], dim=-1)
        h_current = self.gru_unit(gru_input, h_prev)
        return h_current

    def observe(self, z_prev, h_prev, obs_embed, action):
        """Posterior step used during world-model training.

        Returns: z_current, h_current, mu_post, sigma_post, mu_prio, sigma_prio
        """
        h_current = self._step_h(z_prev, h_prev, action)

        post = self.posterior(torch.cat([h_current, obs_embed], dim=-1))
        prio = self.prior(h_current)
        mu_post, sigma_post = self.split_params(post)
        mu_prio, sigma_prio = self.split_params(prio)

        z_current = self.latent_representation(mu_post, sigma_post)
        return z_current, h_current, mu_post, sigma_post, mu_prio, sigma_prio

    def imagine(self, z_prev, h_prev, action):
        """Prior-only step used during latent imagination rollouts.

        Returns: z_current, h_current, mu_prio, sigma_prio
        """
        h_current = self._step_h(z_prev, h_prev, action)
        prio = self.prior(h_current)
        mu_prio, sigma_prio = self.split_params(prio)
        z_current = self.latent_representation(mu_prio, sigma_prio)
        return z_current, h_current, mu_prio, sigma_prio


# ===========================================================================
# Actor (per-agent policy over discrete actions)
# ===========================================================================
class Actor(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128,
                 comm_dim: int = 64, hidden_state: int = 256, action_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim + comm_dim, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, action_dim)
        )

    def forward(self, z, h, c):
        logits = self.net(torch.cat([z, h, c], dim=-1))
        return logits  # logits, NOT probabilities

    def distribution(self, z, h, c):
        return torch.distributions.Categorical(logits=self.forward(z, h, c))


# ===========================================================================
# Continue head: predicts P(episode continues) = 1 - done
# ===========================================================================
class ContinueHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 32,
                 hidden_layer: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, 1)
        )

    def forward(self, z, h):
        return self.net(torch.cat([z, h], dim=-1))  # logit

    def prob(self, z, h):
        return torch.sigmoid(self.forward(z, h))


# ===========================================================================
# Reward head: predicts symlog(reward)
# ===========================================================================
class RewardHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 32,
                 hidden_layer: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, 1)
        )

    def forward(self, z, h):
        return self.net(torch.cat([z, h], dim=-1))
