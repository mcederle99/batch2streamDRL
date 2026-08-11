# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from sparse_init import sparse_init 
from torch.distributions.utils import _standard_normal
from optim import ObGD as Optimizer
from optim import NormSGD
import tyro
import pickle
from normalization_wrappers import ScaleRewardEval, NormalizeObservationEval, SampleMeanStd
from streaming_utils import evaluate
import filecmp
import warnings
warnings.filterwarnings("ignore")


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = "dm_control/walker-run-v0"
    """the id of the environment"""
    gamma: float = 0.99
    """the discount factor gamma"""
    hidden_size : int = 128
    """hidden size MLPs"""
    debug: bool = True
    """whether to print at each episode or not"""
    eval_interval: int = 10000
    """evaluation frequency"""
    total_steps: int = 5000000
    """total number of fine tuning steps"""
    lamda: float = 0.8
    """Trace lambda parameter"""
    kappa_value: float = 2.0
    """Kappa value"""
    actor_lr_den: float = 256.0
    """Learning rate for the actor"""
    actor_lr_optim: str = "adam"
    """Actor learning rate optimizer"""
    target_noise: float = 0.1
    """Target policy smoothing noise"""
    freeze_stats: bool = False
    """whether to freeze obs and reward stats during finetuning"""
    freeze_Q: bool = False
    """whether to freeze Q function during finetuning"""
    Q_optim: str = "obgd"
    """Q function optimizer"""
    starting_checkpoint: str = "500.00K"
    """Starting checkpoint for finetuning (in K or M steps)"""
    pre_training_seed: int = 1
    """Seed used during pre-training phase"""
    checkpoint_dir: str = "adamw-td3-walker-run-checkpoint_1_500.00K.pt"
    """Directory of the checkpoint to finetune from"""
    warmup_steps: int = 5000
    """Number of warmup steps for critic and statistics"""
    exploration_noise: float = 0.1
    """Std of the exploration noise added to the actions during finetuning"""
    mixed: str = "_mixed"
    """Whether to use mixed pre-training checkpoints, use '_' if you want to use Adam checkpoints"""


def set_one_thread():
    '''
    N.B: Pytorch over-allocates resources and hogs CPU, which makes experiments very slow!
    Set number of threads for pytorch to 1 to avoid this issue. This is a temporary workaround.
    '''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def network_weightnorm_l2(model):
    total = 0.0
    for m in model.modules():
        if hasattr(m, "weight_g"):
            total += (m.weight_g ** 2).sum().item()
        elif hasattr(m, "weight"):  # non-weightnorm layers
            total += (m.weight ** 2).sum().item()
    return total ** 0.5


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
                 pnorm=False, actor_lr=3e-4, std_clip=1.0, optim="",
                 target_noise=0.1, exploration_noise=0.1):
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
        self.exploration_noise = exploration_noise

    def sample_action(self, s, eval_=False, target=False):
        s= torch.from_numpy(s).float() if isinstance(s, np.ndarray) else s
        mu = self.policy_net(s)

        if eval_:
            return mu

        std = torch.ones_like(mu) * self.exploration_noise

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
            s_prime = torch.from_numpy(s_prime).float() if isinstance(s_prime, np.ndarray) else s_prime
            mu = self.policy_net(s_prime)
            std = torch.ones_like(mu) * self.target_noise
            dist = TruncatedNormal(mu, std)
            next_action = dist.sample(clip=self.std_clip)
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


    def update_critic(self, s, a, r, s_prime, term, trunc):
        done_mask = 0 if term else 1
        s, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)

        # Q loss
        q_s = self.Q(s, a.detach())

        with torch.no_grad():
            s_prime = torch.from_numpy(s_prime).float() if isinstance(s_prime, np.ndarray) else s_prime
            mu = self.policy_net(s_prime)
            std = torch.ones_like(mu) * self.target_noise
            dist = TruncatedNormal(mu, std)
            next_action = dist.sample(clip=self.std_clip)
            q_s_prime = self.Q(s_prime, next_action)

            td_target = q_s_prime

        delta = r + self.gamma * td_target * done_mask - q_s
        q_output = -q_s

        self.optimizer_value.zero_grad()
        q_output.backward()
        self.optimizer_value.step(delta.item(), reset=term or trunc) 

    def update_policy(self, s):
        s = torch.tensor(np.array(s), dtype=torch.float)
        ploss = - self.Q(s, self.policy_net(s))

        self.optimizer_policy.zero_grad()
        ploss.backward()
        self.optimizer_policy.step()       


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.actor_lr_optim == "adam":
        args.actor_lr = 0.0003 / args.actor_lr_den
    elif args.actor_lr_optim == "nsgd":
        args.actor_lr = 0.25 / args.actor_lr_den
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    checkpoint = torch.load(args.checkpoint_dir, map_location="cpu")

    # same = filecmp.cmp("xmls_links/dog.xml", "xmls_original/dm_control/dog.xml", shallow=False)
    # if same:
    #     print("We are using the original environment")
    # else:
    #     print("We are using a modified environment")
    # same = filecmp.cmp("xmls_links/walker.xml", "xmls_original/dm_control/walker.xml", shallow=False)
    # if same:
    #     print("We are using the original environment")
    # else:
    #     print("We are using a modified environment")

    # env setup
    env = gym.make(args.env_id, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

    # obs/reward norm
    obs_stats = SampleMeanStd(shape=env.observation_space.shape)
    reward_stats = SampleMeanStd(shape=())
    reward_trace = np.zeros(1) 
    # Restore stats
    if "obs_stats" in checkpoint:
        s = checkpoint["obs_stats"]
        obs_stats.mean = s["mean"]
        obs_stats.var = s["var"]
        obs_stats.p = s["p"]
        obs_stats.count = s["count"]

    if "reward_stats" in checkpoint:
        s = checkpoint["reward_stats"]
        reward_stats.mean = s["mean"]
        reward_stats.var = s["var"]
        reward_stats.p = s["p"]
        reward_stats.count = s["count"]
    # ==============
    env = ScaleRewardEval(env, gamma=args.gamma, reward_stats=reward_stats, reward_trace=reward_trace)
    env = NormalizeObservationEval(env, obs_stats=obs_stats)
    
    eval_env = gym.make(args.env_id, render_mode="rgb_array")
    eval_env = gym.wrappers.FlattenObservation(eval_env)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
    eval_env = gym.wrappers.ClipAction(eval_env)
    eval_env = NormalizeObservationEval(eval_env, obs_stats=obs_stats)

    agent = StreamDAC(n_obs=eval_env.observation_space.shape[0],
                      n_actions=eval_env.action_space.shape[0],
                      hidden_size=args.hidden_size,
                      lr=1.0,
                      gamma=args.gamma,
                      lamda=args.lamda,
                      kappa_Q=args.kappa_value,
                      pnorm=False,
                      actor_lr=args.actor_lr,
                      std_clip=1.0,
                      optim=args.actor_lr_optim,
                      target_noise=args.target_noise,
                      exploration_noise=args.exploration_noise)
    
    # LOAD POLICY AND CRITIC WEIGHTS
    _, _ = agent.policy_net.load_state_dict(checkpoint["actor"], strict=False)
    _, _ = agent.Q.load_state_dict(checkpoint["qf1"], strict=False)
    
    # LOAD POLICY OPTIMIZER AND CRITIC OPTIMIZER IF WE ARE USING ADAM
    if args.actor_lr_optim == "adam":
        agent.optimizer_policy.load_state_dict(checkpoint["actor_opt"])
        for param_group in agent.optimizer_policy.param_groups:
            param_group["lr"] = args.actor_lr

    save_dir = f"finetuning_results/{args.env_id}_streamdac_{int(args.kappa_value)}_{args.starting_checkpoint}_{args.pre_training_seed}_{args.actor_lr_optim}_{args.warmup_steps}_{args.exploration_noise}_{args.target_noise}{args.mixed}_{args.actor_lr_den}"
    
    set_one_thread()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.debug:
        print("seed: {}".format(args.seed), "env: {}".format(env.spec.id))

    returns, term_time_steps = [], []

    eval_returns = [evaluate(eval_env, agent, args.seed)]
    eval_runs = 1
    
    if args.warmup_steps > 0:
        args.freeze_stats = True
    s, _ = env.reset(seed=args.seed, freeze_stats=args.freeze_stats)
    for t in range(1, args.total_steps+1):
        if args.exploration_noise == 0.0:
            a = agent.sample_action(torch.from_numpy(s).float(), eval_=True)
        else:
            a = agent.sample_action(torch.from_numpy(s).float(), eval_=False)
        sim_action = a.detach().cpu().numpy()
        s_prime, r, terminated, truncated, info = env.step(sim_action, freeze_stats=args.freeze_stats)
        if args.freeze_Q:
            agent.update_policy(s)
        else:
            if t < args.warmup_steps:
                agent.update_critic(s, a, r, s_prime, terminated, truncated)
            else:    
                agent.update_params(s, a, r, s_prime, terminated, truncated)
                args.freeze_stats = False
        
        s = s_prime
        if terminated or truncated:
            if t > args.eval_interval * eval_runs and t < args.eval_interval * (eval_runs + 1):
                eval_returns.append(evaluate(eval_env, agent, args.seed, obs_stats=env.obs_stats))
                print(f"Average evaluation return across 10 episodes: {eval_returns[-1]} at time step {t}")
                eval_runs += 1
            if args.debug:
                print("Episodic Return: {}, Time Step {}".format(info['episode']['r'][0], t))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset(freeze_stats=args.freeze_stats)

    env.close()

    # Save training data
    with open(os.path.join(save_dir, f"trainreturns_{args.seed}.pkl"), "wb") as f:
        pickle.dump((returns, term_time_steps, args.env_id), f)
    np.save(f"{save_dir}/evalreturns_{args.seed}.npy", eval_returns)
