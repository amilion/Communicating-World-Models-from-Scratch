import torch
from torch.nn import functional as F
import random

try:
    from .loss_functions import masked_mean, symlog, symexp  
except ImportError: 
    from loss_functions import masked_mean, symlog, symexp  


# ===========================================================================
# Episode collection
# ===========================================================================
# Collects ONE aligned record per environment cycle (a cycle = one
# pass over all agents in the AEC loop) using a STABLE agent ordering taken from
# env.possible_agents. For each cycle it stores [N, F] slices and an alive mask.
# Dead/absent agents get zero-filled entries and alive_mask = 0, so the data is
# rectangular and dead agents are cleanly excluded downstream.
#
# Returns a dict of padded tensors (see SequenceBufferReplay.append contract):
#     'obs'   [T, N, obs_dim]   'act'  [T, N, action_dim]
#     'rew'   [T, N]            'done' [T, N]
#     'seq_mask' [T]            'alive_mask' [T, N]
def collect_episode(env, encoders, rssms, communication_head, actors,
                    max_cycles: int = 50, epsilon: float = 0.1, device=None):
    env.reset()
    agents = list(env.possible_agents)
    agent_index = {a: i for i, a in enumerate(agents)}
    N = len(agents)
    action_dims = {a: env.action_space(a).n for a in agents}
    obs_dim = env.observation_space(agents[0]).shape[0]

    _hs = {a: rssms[agent_index[a]].initial_states(1, device)[1] for a in agents}
    _zs = {a: rssms[agent_index[a]].initial_states(1, device)[0] for a in agents}
    _as = {a: torch.zeros(1, action_dims[a], device=device) for a in agents}

    cyc_obs, cyc_act, cyc_rew, cyc_done, cyc_alive = [], [], [], [], []

    def fresh_cycle():
        return (
            torch.zeros(N, obs_dim),
            torch.zeros(N, max(action_dims.values())),
            torch.zeros(N),
            torch.zeros(N),
            torch.zeros(N),
        )

    cur = fresh_cycle()
    seen_this_cycle = set()

    for agent in env.agent_iter():
        i = agent_index[agent]
        obs, rew, term, trunc, _ = env.last()
        done = bool(term or trunc)

        cur[0][i] = torch.as_tensor(obs, dtype=torch.float32)
        cur[2][i] = float(rew)
        cur[3][i] = float(term)              # per-agent terminal (not truncation)
        cur[4][i] = 1.0

        embed = encoders[i](torch.as_tensor(obs, dtype=torch.float32)
                            .reshape(1, -1).to(device))
        if not done:
            with torch.no_grad():
                _zs[agent], _hs[agent], *_ = rssms[i].observe(
                    _zs[agent], _hs[agent], embed, _as[agent])

        with torch.no_grad():
            zs = torch.stack([_zs[a] for a in agents], dim=1)   # [1, N, latent]
            hs = torch.stack([_hs[a] for a in agents], dim=1)   # [1, N, hidden]
            context = communication_head(zs, hs)                # [1, N, comm]
            c_i = context[:, i, :]

        if done:
            action = None
        elif random.random() < epsilon:
            action = env.action_space(agent).sample()
        else:
            with torch.no_grad():
                dist = actors[i].distribution(_zs[agent], _hs[agent], c_i)
                action = int(dist.sample().item())

        env.step(action)

        if action is not None:
            onehot = F.one_hot(torch.tensor(action),
                               num_classes=action_dims[agent]).float()
            _as[agent] = onehot.reshape(1, -1).to(device)
            cur[1][i, :action_dims[agent]] = onehot

        seen_this_cycle.add(agent)
        if seen_this_cycle >= set(env.agents) and len(seen_this_cycle) > 0:
            cyc_obs.append(cur[0]); cyc_act.append(cur[1])
            cyc_rew.append(cur[2]); cyc_done.append(cur[3])
            cyc_alive.append(cur[4])
            cur = fresh_cycle()
            seen_this_cycle = set()
            if len(cyc_obs) >= max_cycles:
                break

    if seen_this_cycle:
        cyc_obs.append(cur[0]); cyc_act.append(cur[1])
        cyc_rew.append(cur[2]); cyc_done.append(cur[3])
        cyc_alive.append(cur[4])

    T = len(cyc_obs)
    episode = {
        "obs":  torch.stack(cyc_obs, dim=0),    # [T, N, obs_dim]
        "act":  torch.stack(cyc_act, dim=0),    # [T, N, action_dim]
        "rew":  torch.stack(cyc_rew, dim=0),    # [T, N]
        "done": torch.stack(cyc_done, dim=0),   # [T, N]
        "seq_mask": torch.ones(T),              # all collected steps are real
        "alive_mask": torch.stack(cyc_alive, dim=0),  # [T, N]
    }
    return episode


# ===========================================================================
# Lambda returns (GAE-style bootstrapped returns for the critic/actor targets)
# ===========================================================================
# Standard Dreamer lambda-return:
#     G_t = r_t + gamma * c_t * [ (1-lam) * V_{t+1} + lam * G_{t+1} ]
# with G_T bootstrapped from the last value estimate. Shapes are [T, B, 1].
def compute_lambda_returns(rewards, values, continues, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    G = values[-1]
    for t in reversed(range(T)):
        boot = (1 - lam) * values[t] + lam * G if t < T - 1 else values[t]
        G = rewards[t] + gamma * continues[t] * boot
        returns[t] = G
    return returns
