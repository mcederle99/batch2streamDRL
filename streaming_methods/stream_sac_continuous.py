import os, pickle, argparse
import torch
import math
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from optim import ObGD as Optimizer
from optim import NormSGD
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, NormalizeObservationEval, ScaleReward
from sparse_init import sparse_init
from streaming_utils import evaluate
import warnings
warnings.filterwarnings("ignore")


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)


def set_one_thread():
    '''
    N.B: Pytorch over-allocates resources and hogs CPU, which makes experiments very slow!
    Set number of threads for pytorch to 1 to avoid this issue. This is a temporary workaround.
    '''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)




class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, pnorm=False, squash=False):
        super(Actor, self).__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)

        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)

        self.apply(initialize_weights)

        self.pnorm = pnorm
        self.squash = squash

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
        log_std = self.linear_std(phi)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        # log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        if not self.squash:
            mu = F.tanh(mu)

        return mu, log_std


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


class StreamSAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_Q=2.0,
                 pnorm=False, squash=False, eta=0.07, actor_lr=3e-4, optim="adam", eta_decay_type="none",
                 eta_lr=1e-4, scale_factor=1000):
        super(StreamSAC, self).__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size, pnorm=pnorm, squash=squash)
        self.Q = Q(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size, pnorm=pnorm)
        self.optimizer_value = Optimizer(self.Q.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_Q)
        if optim == "adam":
            self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=actor_lr, betas=(0.9,0.999))
        elif optim == "nsgd":    
            self.optimizer_policy = NormSGD(self.policy_net.parameters(), lr=actor_lr)
        else:
            raise Exception("No optim valid")

        self.squash = squash
        self.eta_init = eta
        self.eta = eta

        self.eta_decay_type = eta_decay_type

        if self.eta_decay_type == "reward_based":
            self.scale_factor = scale_factor
        elif self.eta_decay_type == "autotune":
            # Automatic entropy tuning
            self.target_entropy = -torch.Tensor([n_actions]).item()
            self.log_eta = torch.tensor(np.log(self.eta_init), requires_grad=True)
            self.eta = self.log_eta.exp().item()
            self.eta_optimizer = torch.optim.Adam([self.log_eta], lr=eta_lr)
        elif self.eta_decay_type != "none":
            raise Exception("No eta decay type valid")


        self.etas = [self.eta_init]

    def sample_action(self, s, eval_=None):
        s= torch.from_numpy(s).float() if isinstance(s, np.ndarray) else s
        mu, log_std = self.policy_net(s)
        if eval_:
            return F.tanh(mu)
        dist = Normal(mu, log_std.exp())

        action_pre = dist.rsample() # for RT
        lprob = dist.log_prob(action_pre)  # log p
        
        if self.squash:
            lprob -= (2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre)))
            action = F.tanh(action_pre)
        else:
            action = action_pre

        return action, lprob.sum(dim=-1)


    def update_params(self, s, a, lprob, r, s_prime, term, trunc, extra=None):


        if self.eta_decay_type == "reward_based":
            eta_ = self.eta_init / (extra + 1e-08)
        elif self.eta_decay_type in ["autotune"]:
            eta_ = self.eta / (extra + 1e-08)
        elif self.eta_decay_type == "none":
            eta_ = self.eta_init
        
        done_mask = 0 if term else 1
        s, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)

        # Q loss
        q_s = self.Q(s, a.detach())

        with torch.no_grad():
            next_action, lprob_next = self.sample_action(s_prime)
            q_s_prime = self.Q(s_prime, next_action)

            td_target = q_s_prime - eta_ * lprob_next

        delta = r + self.gamma * td_target * done_mask - q_s
        q_output = -q_s

        self.optimizer_value.zero_grad()
        q_output.backward()
        self.optimizer_value.step(delta.item(), reset=term or trunc)


        # PI loss
        ploss = eta_ * lprob - self.Q(s, a) 

        self.optimizer_policy.zero_grad()
        ploss.backward()
        self.optimizer_policy.step()

        if self.eta_decay_type == "autotune":
            with torch.no_grad():
                _, lprob = self.sample_action(s)
            eta_loss = (-self.log_eta.exp() * (lprob + self.target_entropy)).mean()

            self.eta_optimizer.zero_grad()
            eta_loss.backward()
            self.eta_optimizer.step()
            self.eta = self.log_eta.exp().item()

            self.etas.append(self.eta)


def main(env_name, seed, lr, gamma, lamda, total_steps, kappa_value, debug,
         time_info=True, eta=0.07, actor_lr=3e-4, squash=False, hidden_size=256,
         optim="adam", eval_interval=50000, eta_decay_type="none", eta_lr=1e-4):
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

    agent = StreamSAC(n_obs=env.observation_space.shape[0],
                    n_actions=env.action_space.shape[0],
                    hidden_size=hidden_size,
                    lr=lr,
                    gamma=gamma, 
                    lamda=lamda, 
                    kappa_Q=kappa_value,
                    pnorm=False,
                    squash=squash,
                    eta=eta,
                    actor_lr=actor_lr,
                    optim=optim,
                    eta_decay_type=eta_decay_type,
                    eta_lr=eta_lr
                    )
        
    save_dir = f"stream_results/streamsac-{env_name}_{eta}_{actor_lr}_{optim}_{eta_decay_type}"
    
    set_one_thread()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))

    returns, term_time_steps = [], []
    std_rs = []

    eval_returns = [evaluate(eval_env, agent, seed)]
    eval_runs = 1

    s, _ = env.reset(seed=seed)

    for t in range(1, total_steps+1):
        a, lprob = agent.sample_action(torch.from_numpy(s).float())
        sim_action = a.detach().cpu().numpy()
        s_prime, r, terminated, truncated, info = env.step(sim_action)
        std_r = env.env.reward_stats.var ** 0.5
        std_rs.append(float(std_r))
        agent.update_params(s, a, lprob, r, s_prime, terminated, truncated, extra=float(std_r))
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
    np.save(f"{save_dir}/etas_{seed}.npy", np.array(agent.etas))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream SAC(λ)')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=20_000_000)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_time_info', action='store_true')
    parser.add_argument('--eta', type=float, default=0.002)
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--squash', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--optim', type=str, choices=["adam", "nsgd"])
    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--eta_decay_type', type=str, choices=["none", "reward_based"])
    ### new additions for autotune
    parser.add_argument('--eta_lr', type=float, default=1e-4)
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
        args.eta,
        args.actor_lr,
        args.squash,
        args.hidden_size,
        args.optim,
        args.eval_interval,
        args.eta_decay_type,
        args.eta_lr
        )
