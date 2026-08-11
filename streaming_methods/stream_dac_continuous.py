import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.utils import _standard_normal
from optim import ObGD as Optimizer, NormSGD
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, NormalizeObservationEval, ScaleReward
from sparse_init import sparse_init
from streaming_utils import evaluate
import warnings
warnings.filterwarnings("ignore")


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


class TruncatedNormal(Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, pnorm=False):
        super(Actor, self).__init__()

        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)

        self.linear_mu = nn.Linear(hidden_size, n_actions)

        self.apply(initialize_weights)

        self.pnorm = pnorm

    def forward(self, x):

        z = self.fc_layer(x)
        z = F.layer_norm(z, z.size())
        z = F.leaky_relu(z)
        z = self.hidden_layer(z)
        z = F.layer_norm(z, z.size()) if not self.pnorm else z
        z = F.leaky_relu(z)

        if self.pnorm:
            if len(z.size) == 1:
                phi = z / torch.norm(z, dim=-1)
            else:
                phi = z / torch.norm(z, dim=-1).view((-1, 1))
        else:
            phi = z

        mu = self.linear_mu(phi)

        return F.tanh(mu)


class Q(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, pnorm=False):
        super(Q, self).__init__()

        self.fc_layer   = nn.Linear(n_obs+n_actions, hidden_size)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)

        self.output_linear = nn.Linear(hidden_size, 1)

        self.apply(initialize_weights)

        self.pnorm = pnorm

    def forward(self, x, action):

        z = torch.cat((x, action), dim=-1)

        z = self.fc_layer(z)
        z = F.layer_norm(z, z.size())
        z = F.leaky_relu(z)
        z = self.hidden_layer(z)
        z = F.layer_norm(z, z.size()) if not self.pnorm else z
        z = F.leaky_relu(z)

        if self.pnorm:
            if len(z.size) == 1:
                phi = z / torch.norm(z, dim=-1)
            else:
                phi = z / torch.norm(z, dim=-1).view((-1, 1))
        else:
            phi = z

        return self.output_linear(phi)



class StreamDAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_Q=2.0,
                 pnorm=False, actor_lr=3e-4, std_clip=1.0,  optim="adam", target_noise="n"):
        super(StreamDAC, self).__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size, pnorm=pnorm)
        self.Q = Q(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size, pnorm=pnorm)
        self.optimizer_value = Optimizer(self.Q.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_Q)
        if optim == "adam":
            self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=actor_lr, betas=(0.9,0.999))
        elif optim == "nsgd":
            self.optimizer_policy = NormSGD(self.policy_net.parameters(), lr=actor_lr)
        else:
            raise Exception("No optim valid")

        self.std_clip = std_clip
        self.target_noise = target_noise


    def sample_action(self, s, eval_=False):
        s= torch.from_numpy(s).float() if isinstance(s, np.ndarray) else s
        mu = self.policy_net(s)

        if eval_:
            return mu

        std = torch.ones_like(mu) * 0.2

        dist = TruncatedNormal(mu, std)

        action = dist.sample(clip=self.std_clip)

        return action

    def update_params(self, s, a, r, s_prime, term, trunc):

        done_mask = 0 if term else 1
        s, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)

        # Q loss
        q_s = self.Q(s, a.detach())

        with torch.no_grad():

            if self.target_noise == "y":
                s_prime = torch.from_numpy(s_prime).float() if isinstance(s_prime, np.ndarray) else s_prime
                mu = self.policy_net(s_prime)
                std = torch.ones_like(mu) * 0.2
                dist = TruncatedNormal(mu, std)
                next_action = dist.sample(clip=self.std_clip)
            else:
                s_prime = torch.from_numpy(s_prime).float() if isinstance(s_prime, np.ndarray) else s_prime
                next_action = self.policy_net(s)
            q_s_prime = self.Q(s_prime, next_action)

            td_target = q_s_prime

        delta = r + self.gamma * td_target * done_mask - q_s
        q_output = -q_s

        self.optimizer_value.zero_grad()
        q_output.backward()
        self.optimizer_value.step(delta.item(), reset=term or trunc)

        # PI loss
        ploss = - self.Q(s, self.policy_net(s))

        self.optimizer_policy.zero_grad()
        ploss.backward()
        self.optimizer_policy.step()


def main(env_name, seed, lr, gamma, lamda, total_steps, kappa_value, debug,
         time_info=True, std_clip=1.0, actor_lr=3e-4, hidden_size=128,
           eval_interval=50000, optim="adam", target_noise="n"):
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


    agent = StreamDAC(n_obs=env.observation_space.shape[0],
                    n_actions=env.action_space.shape[0],
                    hidden_size=hidden_size,
                    lr=lr,
                    gamma=gamma, 
                    lamda=lamda, 
                    kappa_Q=kappa_value,
                    pnorm=False,
                    actor_lr=actor_lr,
                    std_clip=std_clip,
                    optim=optim,
                    target_noise=target_noise
                    )
        
    save_dir = f"stream_results/streamdac-{env_name}_{actor_lr}_{optim}_{target_noise}"
    
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
        a = agent.sample_action(torch.from_numpy(s).float())
        sim_action = a.detach().cpu().numpy()
        s_prime, r, terminated, truncated, info = env.step(sim_action)
        agent.update_params(s, a, r, s_prime, terminated, truncated)
        s = s_prime
        if terminated or truncated:
            if t > eval_interval * eval_runs and t < eval_interval * (eval_runs + 1):
                eval_returns.append(evaluate(eval_env, agent, seed, obs_stats=env.obs_stats))
                print(f"Average evaluation return across 10 episodes: {eval_returns[-1]} at time step {t}")
                eval_runs += 1
            if debug:
                print("Episodic Return: {}, Time Step {}".format(info['episode']['r'][0], t))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset()

    env.close()
    # Save training data
    with open(os.path.join(save_dir, f"trainreturns_{seed}.pkl"), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)
    np.save(f"{save_dir}/evalreturns_{seed}.npy", eval_returns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream DAC(λ)')
    parser.add_argument('--env_name', type=str, default="HalfCheetah-v4")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=20_000_000)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_time_info', action='store_true')
    parser.add_argument('--std_clip', type=float, default=1.0)
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--optim', type=str, choices=["adam", "nsgd"])
    parser.add_argument('--target_noise', type=str, choices=["y", "n"])
    args = parser.parse_args()
    main(args.env_name,
        args.seed,
        args.lr,
        args.gamma, 
        args.lamda, 
        args.total_steps,   
        args.kappa_value, 
        args.debug, 
        not args.no_time_info,
        args.std_clip,
        args.actor_lr,
        args.hidden_size,
        args.eval_interval,
        args.optim,
        args.target_noise
        )
