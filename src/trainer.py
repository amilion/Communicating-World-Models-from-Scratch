import copy
import numpy as np
import torch
import torch.nn.functional as F

try:  
    from .buffer import SequenceBufferReplay
    from .VAE import Encoder, Decoder
    from .WM import RSSM, Actor, ContinueHead, RewardHead
    from .centerallize_behavior import Critic
    from .helpers import collect_episode, compute_lambda_returns
    from .loss_functions import symlog, kl_loss, masked_mean, compute_temporal_curvature
except ImportError:
    from buffer import SequenceBufferReplay
    from VAE import Encoder, Decoder
    from WM import RSSM, Actor, ContinueHead, RewardHead
    from centerallize_behavior import Critic
    from helpers import collect_episode, compute_lambda_returns
    from loss_functions import symlog, kl_loss, masked_mean, compute_temporal_curvature


# ===========================================================================
# World-model training
# ===========================================================================
def train_world_model(batch, rssms, encoders, decoders, reward_heads,
                      continue_heads, wm_optimizers, n_agents,
                      free_bits=1.0, kl_scale=1.0, grad_clip=100.0,
                      lambda_curv=0.05, debug=False):
    """One masked world-model update per agent.

    `batch` is the dict returned by SequenceBufferReplay.sample:
        obs        [B, T, N, obs_dim]
        act        [B, T, N, action_dim]
        rew        [B, T, N]
        done       [B, T, N]
        seq_mask   [B, T]
        alive_mask [B, T, N]

    Returns: (mean_loss, z_start, h_start) where z_start/h_start are the
    POSTERIOR states at EVERY (b, t) -- flattened to [B*T, N, F] -- so the
    behaviour step can imagine from the full state distribution (AUDIT A10),
    not just t == 0.
    """
    obs = batch["obs"].float()
    act = batch["act"].float()
    rew = batch["rew"].float()
    done = batch["done"].float()
    seq_mask = batch["seq_mask"].float()       # [B, T]
    alive_mask = batch["alive_mask"].float()   # [B, T, N]

    B, T, N, _ = obs.shape
    device = obs.device

    total_loss = 0.0
    total_curv_loss = 0.0
    z_states = [None] * n_agents
    h_states = [None] * n_agents

    diag = {}
    for i in range(n_agents):
        z_prev, h_prev = rssms[i].initial_states(B, device)

        rec_l = rew_l = cont_l = kl_l = 0.0
        z_seq, h_seq, mask_seq = [], [], []

        for t in range(T):
            m = (seq_mask[:, t] * alive_mask[:, t, i]).unsqueeze(-1)  # [B, 1]

            obs_t = obs[:, t, i, :]
            act_t = act[:, t, i, :]

            obs_embed = encoders[i](obs_t)
            (z_cur, h_cur, mu_post, sigma_post,
             mu_prio, sigma_prio) = rssms[i].observe(z_prev, h_prev, obs_embed, act_t)

            recon = decoders[i](z_cur, h_cur)
            rew_pred = reward_heads[i](z_cur, h_cur)              # [B, 1]
            cont_logit = continue_heads[i](z_cur, h_cur)         # [B, 1] logit

            rec_l += masked_mean((recon - obs_t) ** 2, m)
            rew_l += masked_mean((rew_pred - symlog(rew[:, t, i:i+1])) ** 2, m)
            cont_target = 1.0 - done[:, t, i:i+1]
            cont_l += masked_mean(
                F.binary_cross_entropy_with_logits(
                    cont_logit, cont_target, reduction="none"), m)
            kl_elem = kl_loss(mu_post, sigma_post, mu_prio, sigma_prio,
                              free_bits=free_bits)                # [B]
            kl_l += masked_mean(kl_elem.unsqueeze(-1), m)

            z_seq.append(z_cur)
            h_seq.append(h_cur)
            mask_seq.append(seq_mask[:, t] * alive_mask[:, t, i])   # [B]

            keep = alive_mask[:, t, i].unsqueeze(-1)             # [B, 1]
            z_prev = z_cur * keep
            h_prev = h_cur * keep
        z_seq_tensor  = torch.stack(z_seq,     dim=1)   # [B, T, latent_dim]
        mask_tensor   = torch.stack(mask_seq,  dim=1)   # [B, T]
        curv_loss = compute_temporal_curvature(z_seq_tensor, mask_tensor)

        loss = ((rec_l + rew_l + cont_l + kl_scale * kl_l) / max(T, 1)
                + lambda_curv * curv_loss)

        wm_optimizers[i].zero_grad()
        loss.backward()
        params = (list(rssms[i].parameters()) + list(encoders[i].parameters())
                  + list(decoders[i].parameters())
                  + list(reward_heads[i].parameters())
                  + list(continue_heads[i].parameters()))
        gnorm = torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        wm_optimizers[i].step()

        assert torch.isfinite(loss), f"non-finite WM loss for agent {i}"

        total_loss      += loss.item()
        total_curv_loss += curv_loss.item()
        z_states[i] = z_seq_tensor                     # [B, T, latent]
        h_states[i] = torch.stack(h_seq, dim=1)        # [B, T, hidden]
        if debug:
            diag[f"agent{i}"] = dict(rec=rec_l.item(), rew=rew_l.item(),
                                     cont=cont_l.item(), kl=kl_l.item(),
                                     curv=curv_loss.item(), grad=float(gnorm))

    z_start = torch.stack(z_states, dim=2).reshape(B * T, N, -1).detach()
    h_start = torch.stack(h_states, dim=2).reshape(B * T, N, -1).detach()

    start_alive = alive_mask.reshape(B * T, N).detach()

    if debug:
        print(f"[WM] loss={total_loss / n_agents:.4f} "
              f"curv={total_curv_loss / n_agents:.4f} "
              f"alive_frac={alive_mask.mean().item():.3f} {diag}")

    return total_loss / n_agents, total_curv_loss / n_agents, z_start, h_start, start_alive


# ===========================================================================
# Behaviour step: latent imagination actor-critic
# ===========================================================================
def behavior_step(rssms, comm_module, actors, critic, target_critic,
                  reward_heads, continue_heads, z_start, h_start, start_alive,
                  actor_opts, critic_opt, horizon=15, gamma=0.99, lam=0.95,
                  entropy_coef=1e-3, grad_clip=100.0, ema_tau=0.02, debug=False):
    """Imagine `horizon` steps from the detached start states and update the
    actor(s) and the centralised critic.
    """
    Bf, n_agents, _ = z_start.shape   # Bf = B*T flattened imagination batch
    device = z_start.device

    current_z, current_h = z_start, h_start
    alive = start_alive.clone()                       # [Bf, N]

    all_log_probs = [[] for _ in range(n_agents)]
    all_entropy = [[] for _ in range(n_agents)]
    all_rewards, all_values, all_cont, all_alive = [], [], [], []

    for t in range(horizon):
        context = comm_module(current_z, current_h)   # [Bf, N, comm]

        per_agent_rew = torch.stack(
            [reward_heads[i](current_z[:, i, :], current_h[:, i, :])
             for i in range(n_agents)], dim=1)        # [Bf, N, 1]
        per_agent_cont = torch.stack(
            [continue_heads[i].prob(current_z[:, i, :], current_h[:, i, :])
             for i in range(n_agents)], dim=1)        # [Bf, N, 1]

        am = alive.unsqueeze(-1)                       # [Bf, N, 1]
        joint_rew = masked_mean_keepbatch(per_agent_rew, am)   # [Bf, 1]
        joint_cont = (per_agent_cont * am).amax(dim=1)          # [Bf, 1]

        value = critic(current_z, current_h, alive_mask=alive)  # [Bf, 1]

        all_rewards.append(joint_rew)
        all_values.append(value)
        all_cont.append(joint_cont)
        all_alive.append(alive.clone())

        new_z, new_h = [], []
        for i in range(n_agents):
            dist = actors[i].distribution(
                current_z[:, i, :], current_h[:, i, :], context[:, i, :])
            a = dist.sample()
            all_log_probs[i].append(dist.log_prob(a))     # [Bf]
            all_entropy[i].append(dist.entropy())         # [Bf]
            a_onehot = F.one_hot(a, num_classes=rssms[i].action_dim).float()
            n_z, n_h, _, _ = rssms[i].imagine(
                current_z[:, i, :], current_h[:, i, :], a_onehot)
            new_z.append(n_z)
            new_h.append(n_h)

        current_z = torch.stack(new_z, dim=1)   # [Bf, N, latent]
        current_h = torch.stack(new_h, dim=1)   # [Bf, N, hidden]
        alive = alive * per_agent_cont.squeeze(-1).detach()

    rewards = torch.stack(all_rewards)        # [H, Bf, 1]
    values = torch.stack(all_values)          # [H, Bf, 1]
    cont = torch.stack(all_cont)              # [H, Bf, 1]

    returns = compute_lambda_returns(rewards, values.detach(), cont, gamma, lam)

    advantage = (returns - values).detach()   # [H, Bf, 1]
    adv_std = advantage.std()
    advantage = (advantage - advantage.mean()) / (adv_std + 1e-8)

    actor_losses = []
    for i in range(n_agents):
        actor_opts[i].zero_grad()
        log_p = torch.stack(all_log_probs[i]).unsqueeze(-1)   # [H, Bf, 1]
        ent = torch.stack(all_entropy[i]).unsqueeze(-1)       # [H, Bf, 1]
        agent_alive = torch.stack(all_alive)[..., i:i+1]      # [H, Bf, 1]
        pg = -(log_p * advantage)                             
        loss_i = masked_mean(pg, agent_alive) \
            - entropy_coef * masked_mean(ent, agent_alive)
        loss_i.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(actors[i].parameters(), grad_clip)
        actor_opts[i].step()
        assert torch.isfinite(loss_i), f"non-finite actor loss agent {i}"
        actor_losses.append(loss_i.item())

    any_alive = torch.stack(all_alive).amax(dim=-1, keepdim=True)  # [H, Bf, 1]
    critic_opt.zero_grad()
    critic_loss = masked_mean((values - returns.detach()) ** 2, any_alive)
    critic_loss.backward()
    grad_params = list(critic.parameters()) + list(comm_module.parameters())
    torch.nn.utils.clip_grad_norm_(grad_params, grad_clip)
    critic_opt.step()
    assert torch.isfinite(critic_loss), "non-finite critic loss"

    with torch.no_grad():
        for p, tp in zip(critic.parameters(), target_critic.parameters()):
            tp.mul_(1 - ema_tau).add_(ema_tau * p)

    if debug:
        print(f"[BEH] critic={critic_loss.item():.4f} "
              f"actor={np.mean(actor_losses):.4f} adv_std={adv_std.item():.4f} "
              f"ret={returns.mean().item():.3f}")

    return critic_loss.item(), float(np.mean(actor_losses))


def masked_mean_keepbatch(x, mask):
    """Masked mean over the AGENT dim only, keeping the batch dim.

    x: [Bf, N, 1], mask: [Bf, N, 1] -> [Bf, 1].
    """
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


# ===========================================================================
# Full training loop
# ===========================================================================
def train_multiagent_dreamer(comm_module, n_agents=2, steps=10000,
                             max_cycles=50, batch_size=16, warmup=8,
                             horizon=15, lambda_curv=0.05,
                             device="cpu", debug=False, seed=0):
    """Collect episodes, train the world model, then the behaviour policy.

    `warmup` = minimum #episodes in the buffer before training begins.
    """
    try:
        from .helpers import collect_episode as _collect
    except ImportError:
        from helpers import collect_episode as _collect

    torch.manual_seed(seed)

    from mpe2 import simple_spread_v3
    env = simple_spread_v3.env(N=n_agents, local_ratio=0.5,
                               max_cycles=max_cycles, render_mode=None)
    env.reset(seed=seed)
    agents = list(env.possible_agents)

    encoders = [Encoder(obs_dim=env.observation_space(a).shape[0]).to(device)
                for a in agents]
    decoders = [Decoder(obs_dim=env.observation_space(a).shape[0]).to(device)
                for a in agents]
    reward_heads = [RewardHead().to(device) for _ in agents]
    continue_heads = [ContinueHead().to(device) for _ in agents]
    rssms = [RSSM(action_dim=env.action_space(a).n).to(device) for a in agents]
    actors = [Actor(action_dim=env.action_space(a).n).to(device) for a in agents]
    comm_module = comm_module.to(device)
    critic = Critic(n_agents=n_agents).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    for p in target_critic.parameters():
        p.requires_grad_(False)

    wm_optims = [torch.optim.Adam(
        list(rssms[i].parameters()) + list(encoders[i].parameters())
        + list(decoders[i].parameters()) + list(reward_heads[i].parameters())
        + list(continue_heads[i].parameters()), lr=1e-3)
        for i in range(n_agents)]
    actor_opts = [torch.optim.Adam(actors[i].parameters(), lr=1e-3)
                  for i in range(n_agents)]
    critic_opt = torch.optim.Adam(
        list(comm_module.parameters()) + list(critic.parameters()), lr=1e-3)

    replay_buffer = SequenceBufferReplay()
    wm_losses, curv_losses, critic_losses, actor_losses, ep_rewards = [], [], [], [], []

    for step in range(steps):
        episode = _collect(env, encoders, rssms, comm_module, actors,
                           max_cycles=max_cycles, device=device)
        replay_buffer.append(episode)
        ep_rewards.append((episode["rew"] * episode["alive_mask"]).sum().item()
                          / episode["alive_mask"].sum().clamp_min(1).item())

        if len(replay_buffer) >= warmup:
            batch = replay_buffer.sample(batch_size)
            batch = {k: v.to(device) for k, v in batch.items()}
            wm_loss, curv_loss_val, z_start, h_start, start_alive = train_world_model(
                batch, rssms, encoders, decoders, reward_heads,
                continue_heads, wm_optims, n_agents,
                lambda_curv=lambda_curv, debug=debug)
            wm_losses.append(wm_loss)
            curv_losses.append(curv_loss_val)

            cri_loss, act_loss = behavior_step(
                rssms, comm_module, actors, critic, target_critic,
                reward_heads, continue_heads, z_start, h_start, start_alive,
                actor_opts, critic_opt, horizon=horizon, debug=debug)
            critic_losses.append(cri_loss)
            actor_losses.append(act_loss)

        if step % 50 == 0:
            mean_r = np.mean(ep_rewards[-50:]) if ep_rewards else 0.0
            print(f"Step {step}/{steps} | Reward: {mean_r:.3f} | "
                  f"buffer={len(replay_buffer)}")

    env.close()
    return (
        wm_losses, curv_losses, critic_losses, actor_losses, ep_rewards,
        encoders, rssms, decoders, continue_heads, reward_heads,
        actors, critic, comm_module,
    )

if __name__ == "__main__":
    from centerallize_behavior import NoCommunication
    train_multiagent_dreamer(NoCommunication(), n_agents=3, steps=12,
                             max_cycles=10, batch_size=4, warmup=3,
                             horizon=4, debug=True)
