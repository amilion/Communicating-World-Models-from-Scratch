import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# symlog / symexp
# ---------------------------------------------------------------------------
# The correct DreamerV3 symlog compresses the target so that large-magnitude
# rewards/values stay in a trainable range:
#     symlog(x) = sign(x) * log(1 + |x|)
#     symexp(x) = sign(x) * (exp(|x|) - 1)   # exact inverse, used to decode
#
# We use `log1p` / `expm1` for numerical stability near 0.
def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


# ---------------------------------------------------------------------------
# KL loss (RSSM dynamics/representation balancing)
# ---------------------------------------------------------------------------
# Correct DreamerV3 KL balancing uses ONE KL -- KL(posterior || prior) --
# computed twice with stop-gradient on alternating sides:
#
#     L_dyn = KL( sg(post) || prior )   # trains the prior towards the post
#     L_rep = KL( post || sg(prior) )   # trains the post towards the prior
#     L_kl  = alpha * L_dyn + (1 - alpha) * L_rep
#
# free_bits clamps each term (per batch element) from below so the model is not
# penalised for already-small KL (prevents over-regularisation / collapse).
def kl_divergence(mu_q, sigma_q, mu_p, sigma_p, eps=1e-8):
    """KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ) summed over the feature dim.

    All args are diagonal-Gaussian parameters of shape [..., latent_dim].
    Returns a tensor of shape [...] (feature dim reduced).
    """
    return (
        torch.log(sigma_p + eps) - torch.log(sigma_q + eps)
        + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2 + eps)
        - 0.5
    ).sum(-1)


def kl_loss(mu_post, sigma_post, mu_prio, sigma_prio,
            alpha=0.8, free_bits=1.0):
    """Balanced KL with stop-gradient, returned PER-ELEMENT (not reduced).

    Args:
        mu_post, sigma_post : posterior params, shape [..., latent_dim]
        mu_prio, sigma_prio : prior params,     shape [..., latent_dim]
        alpha     : weight on the dynamics (prior-training) term.
        free_bits : nats below which the KL is not penalised (per term).

    Returns:
        Tensor of shape [...] (the leading dims), ready to be masked + meaned
        by the caller.
    """
    kl_dyn = kl_divergence(mu_post.detach(), sigma_post.detach(),
                           mu_prio, sigma_prio)
    kl_rep = kl_divergence(mu_post, sigma_post,
                           mu_prio.detach(), sigma_prio.detach())

    kl_dyn = torch.clamp(kl_dyn, min=free_bits)
    kl_rep = torch.clamp(kl_rep, min=free_bits)

    return alpha * kl_dyn + (1.0 - alpha) * kl_rep


# ---------------------------------------------------------------------------
# Temporal Straightening regularisation (Wang et al., 2026)
# ---------------------------------------------------------------------------
# Penalises curvature in each agent's latent trajectory so that z_t evolves
# in locally straight lines.  Straight trajectories align latent geometry
# with environment dynamics, making cross-agent attention more meaningful and
# improving multi-agent scaling.
def compute_temporal_curvature(z_seq, validity_mask, eps=1e-8):
    """Temporal straightening loss for one agent's latent sequence.

    Args:
        z_seq         : [B, T, latent_dim]  stochastic posterior states
        validity_mask : [B, T]              1 where (batch, time) is real
                                            and the agent is alive
        eps           : guard against 0/0 when all triples are masked out

    Returns:
        Differentiable scalar curvature loss.
    """
    T = z_seq.shape[1]
    if T < 3:
        return z_seq.sum() * 0.0

    v = z_seq[:, 1:, :] - z_seq[:, :-1, :]       # [B, T-1, D]  latent velocities
    v_early = v[:, :-1, :]                         # [B, T-2, D]
    v_late  = v[:, 1:,  :]                         # [B, T-2, D]

    cos_sim   = F.cosine_similarity(v_early, v_late, dim=-1)  # [B, T-2]
    curvature = 1.0 - cos_sim                                 # [B, T-2]

    triple_mask = (validity_mask[:, :-2]
                   * validity_mask[:, 1:-1]
                   * validity_mask[:, 2:])          # [B, T-2]

    return (curvature * triple_mask).sum() / (triple_mask.sum() + eps)


# ---------------------------------------------------------------------------
# masked_mean: the single reduction used by EVERY masked loss
# ---------------------------------------------------------------------------
def masked_mean(x, mask):
    """Mean of `x` over the elements where `mask` is nonzero.

    `mask` is broadcast to the shape of `x`. Returns a scalar tensor.
    """
    mask = mask.expand_as(x)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)
