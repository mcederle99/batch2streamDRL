"""Utils for periodic evaluation"""

import numpy as np
import torch


def evaluate_td3(actor, env, step, device, seed, obs_stats = None, num_ep = 10):

    print("---------------------------------------")
    print(f"Evaluation at {step} time steps") 

    returns = np.zeros(num_ep)
    actor.eval()

    for ep in range(num_ep):
        obs, _ = env.reset() if ep > 0 else env.reset(seed=seed)

        while True:
            obs_to_agent = obs_stats.normalize(obs) if obs_stats is not None else torch.Tensor(obs).to(device)
            with torch.no_grad():
                action = actor(obs_to_agent).squeeze(dim=0)  # maybe not necessary TODO

            action_clipped = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)
            obs, _, terminated, truncated, info = env.step(action_clipped)

            if terminated or truncated:
                returns[ep] = info['episode']['r'][0]
                obs, _ = env.reset()
                break
    avg_return = np.mean(returns)
    print(f"Avererage return: {avg_return} - global_steps {step}")
    print("---------------------------------------")

    actor.train()

    return avg_return


def evaluate_sac(actor, env, step, device, seed, obs_stats = None, num_ep = 10, deterministic=False):

    print("---------------------------------------")
    print(f"Evaluation at {step} time steps") 

    returns = np.zeros(num_ep)
    actor.eval()

    for ep in range(num_ep):
        obs, _ = env.reset() if ep > 0 else env.reset(seed=seed)

        while True:
            obs_to_agent = obs_stats.normalize(obs) if obs_stats is not None else torch.Tensor(obs).to(device).unsqueeze(dim=0)
            with torch.no_grad():
                
                if deterministic:
                    action = actor.deterministic_action(obs_to_agent)
                else:
                    action, _, _ = actor.get_action(obs_to_agent)

            action_clipped = action.squeeze(dim=0).cpu().numpy().clip(env.action_space.low, env.action_space.high)
            obs, _, terminated, truncated, info = env.step(action_clipped)

            if terminated or truncated:
                returns[ep] = info['episode']['r'][0]
                obs, _ = env.reset()
                break
    avg_return = np.mean(returns)
    print(f"Avererage return: {avg_return} - global_steps {step}")
    print("---------------------------------------")

    actor.train()

    return avg_return

