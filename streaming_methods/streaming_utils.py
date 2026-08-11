import numpy as np


def evaluate(env, agent, seed, eval_episodes=10, obs_stats=None):

    returns = 0.0

    if obs_stats is not None:
        try:
            env.manually_modify_stats(obs_stats.mean, obs_stats.var, obs_stats.count, obs_stats.p)
        except:
            env.env.manually_modify_stats(obs_stats.mean, obs_stats.var, obs_stats.count, obs_stats.p)

    s, _ = env.reset(seed=seed + 100)
    for _ in range(eval_episodes):
        while True:
            actor_output = agent.sample_action(s, eval_=True)
            a = actor_output[0] if isinstance(actor_output, tuple) else actor_output
            a_sim = a if isinstance(a, np.ndarray) else a.detach().cpu().numpy()

            s, _, terminated, truncated, info = env.step(a_sim)
            
            if terminated or truncated:
                returns += info['episode']['r'][0]
                s, _ = env.reset()
                break
    
    return returns / eval_episodes


def evaluate_avg(env, agent, seed, eval_episodes=10, obs_stats=None):

    returns = []

    if obs_stats is not None:
        try:
            env.manually_modify_stats(obs_stats.mean, obs_stats.var, obs_stats.count, obs_stats.p)
        except:
            env.env.manually_modify_stats(obs_stats.mean, obs_stats.var, obs_stats.count, obs_stats.p)

    s, _ = env.reset(seed=seed + 100)
    for _ in range(eval_episodes):
        ep_returns = 0.0
        while True:
            actor_output, _ = agent.compute_action(s, eval_=True)
            a = actor_output[0] if isinstance(actor_output, tuple) else actor_output
            a_sim = a if isinstance(a, np.ndarray) else a.detach().cpu().view(-1).numpy()

            s, r, terminated, truncated, info = env.step(a_sim)
            
            ep_returns += r
            if terminated or truncated:
                s, _ = env.reset()
                returns.append(ep_returns)
                break
    
    return np.mean(returns)
