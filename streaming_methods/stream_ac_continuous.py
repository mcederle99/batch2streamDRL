import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from optim import ObGD as Optimizer
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, NormalizeObservationEval, ScaleReward
from sparse_init import sparse_init
from streaming_utils import evaluate
import warnings
warnings.filterwarnings("ignore")


def format_steps(steps: int) -> str:
    if steps >= 1_000_000_000:
        return f"{steps / 1_000_000_000:.2f}G"
    elif steps >= 1_000_000:
        return f"{steps / 1_000_000:.2f}M"
    elif steps >= 1_000:
        return f"{steps / 1_000:.2f}K"
    else:
        return str(steps)


def set_one_thread():
    '''
    N.B: Pytorch over-allocates resources and hogs CPU, which makes experiments very slow!
    Set number of threads for pytorch to 1 to avoid this issue. This is a temporary workaround.
    '''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_size=128):
        super(Critic, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
        self.linear_layer  = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)      
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.linear_layer(x)

class StreamAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.value_net = Critic(n_obs=n_obs, hidden_size=hidden_size)
        self.optimizer_policy = Optimizer(self.policy_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = Optimizer(self.value_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def pi(self, x):
        return self.policy_net(x)

    def v(self, x):
        return self.value_net(x)

    def sample_action(self, s, eval_=False):
        x = torch.from_numpy(s).float()
        mu, std = self.pi(x)
        if eval_:
            return mu
        dist = Normal(mu, std)
        return dist.sample().numpy()

    def update_params(self, s, a, r, s_prime, term, trunc, entropy_coeff, overshooting_info=False, time_info=True):
        if time_info:
            done_mask = 0 if term or trunc else 1
        else:
            done_mask = 0 if term else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor(np.array(a)), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)

        v_s, v_prime = self.v(s), self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        mu, std = self.pi(s)
        dist = Normal(mu, std)

        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        self.optimizer_policy.step(delta.item(), reset=term or trunc)
        self.optimizer_value.step(delta.item(), reset=term or trunc)

        if overshooting_info:
            v_s, v_prime = self.v(s), self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")

def main(env_name, seed, lr, gamma, lamda, total_steps, entropy_coeff, kappa_policy, kappa_value, debug, overshooting_info, eval_interval=50000, time_info=True):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)

    eval_env = gym.make(env_name, render_mode="rgb_array")
    eval_env = gym.wrappers.FlattenObservation(eval_env)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
    eval_env = gym.wrappers.ClipAction(eval_env)
    eval_env = NormalizeObservationEval(eval_env)

    if time_info:
        # disable use of time step in state
        env = AddTimeInfo(env)
        eval_env = AddTimeInfo(eval_env)

    agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)

    save_dir = f"stream_results/streamac-{env_name}_{'notimeinfo' if not time_info else 'timeinfo'}"
    
    set_one_thread()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))

    returns, term_time_steps = [], []

    eval_returns = [evaluate(eval_env, agent, seed)]
    eval_runs = 1

    s, _ = env.reset(seed=seed)
    for t in range(1, total_steps+1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(s, a, r, s_prime, terminated, truncated, entropy_coeff, overshooting_info, time_info)
        s = s_prime
        if t % 200000 == 0 and t > 0:
            torch.save(agent.policy_net.state_dict(), f"{save_dir}/policy_{seed}_{format_steps(t)}.pt")
            torch.save(agent.value_net.state_dict(), f"{save_dir}/critic_{seed}_{format_steps(t)}.pt")
        if terminated or truncated:
            if t > eval_interval * eval_runs and t < eval_interval * (eval_runs + 1):
                eval_returns.append(evaluate(eval_env, agent, seed, obs_stats=env.obs_stats if not time_info else env.env.obs_stats))
                print(f"Average evaluation return across 10 episodes: {eval_returns[-1]} at time step {t}")
                eval_runs += 1
            if debug:
                print("Episodic Return: {}, Time Step {}".format(info['episode']['r'][0], t))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset()

    env.close()
    # Save hyper-parameters and config info
    with open(os.path.join(save_dir, f"trainreturns_{seed}.pkl"), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)
    np.save(f"{save_dir}/evalreturns_{seed}.npy", eval_returns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=20_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--no_time_info', action='store_true')
    args = parser.parse_args()
    main(args.env_name,
        args.seed,
        args.lr,
        args.gamma,
        args.lamda,
        args.total_steps,
        args.entropy_coeff,
        args.kappa_policy,
        args.kappa_value,
        args.debug,
        args.overshooting_info,
        args.eval_interval,
        not args.no_time_info)
