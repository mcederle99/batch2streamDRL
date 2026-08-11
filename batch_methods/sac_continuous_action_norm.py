"""Code modified from CleanRL repo: https://github.com/vwxyzjn/cleanrl/tree/master"""
import os
import random
import time
from dataclasses import dataclass
import datetime
import math

import gymnasium as gym
import shimmy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

from buffers import ReplayBuffer
from normalization_utils import SampleMeanStd, SampleStd
from eval_utils import evaluate_sac
from checkpoints_utils import save_sac_checkpoint, load_sac_checkpoint, log_checkpoint_stats, restore_logs

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    eval_freq: int = 10_000
    """evaluation frequency: 10 evaluation episodes"""
    end_checkpoint: bool = False
    """if toggled, actor, q1, q2 and optimizers are saved (last training step)"""
    checkpoint_freq: int = 0
    "if > 0, checkpoint saved every N steps (must be multiple of eval_freq)"
    resume_training: str = ""
    """ path to checkpoint. NOTE: resume is not ideal, we do not restore the target network. therefore, after resuming we have the target nets
        equal to the actual nets
    """

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 3000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 1.0
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    norm_obs: bool = False
    """normalize obs"""
    norm_reward: bool = False
    """normalize reward"""
    use_ln : bool = False
    """use layer norm"""
    hidden_size : int = 128
    """hidden size MLPs"""
    deterministic_eval : bool = True
    """evaluate deterministic policy (no sampling)"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, use_ln=False, hidden_size=128):
        super().__init__()
        self.fc_layer = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            hidden_size,
        )
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, 1)
        self.use_ln = use_ln

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        if not self.use_ln:
            x = F.relu(self.fc_layer(x))
            x = F.relu(self.hidden_layer(x))
        else:
            squeeze = x.dim() == 1

            if squeeze:
                x = x.unsqueeze(dim=0)
                
            x = self.fc_layer(x)
            x = F.layer_norm(x, (x.shape[-1],))
            x = F.leaky_relu(x)
            x = self.hidden_layer(x)
            x = F.layer_norm(x, (x.shape[-1],))
            x = F.leaky_relu(x)

        x = self.output_linear(x)

        if self.use_ln and squeeze:
            x = x.squeeze(dim=0)

        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, use_ln=False, hidden_size=128):
        super().__init__()
        self.fc_layer = nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, np.prod(env.single_action_space.shape))
        self.linear_std = nn.Linear(hidden_size, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.use_ln = use_ln

    def forward(self, x):
        if not self.use_ln:
            x = F.relu(self.fc_layer(x))
            x = F.relu(self.hidden_layer(x))
        else:
            squeeze = x.dim() == 1

            if squeeze:
                x = x.unsqueeze(dim=0)

            x = self.fc_layer(x)
            x = F.layer_norm(x, (x.shape[-1],))
            x = F.leaky_relu(x)
            x = self.hidden_layer(x)
            x = F.layer_norm(x, (x.shape[-1],))
            x = F.leaky_relu(x)

        if self.use_ln and squeeze:
            x = x.squeeze(dim=0)

        mean = self.linear_mu(x)
        log_std = self.linear_std(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def deterministic_action(self, x):
        mean, _ = self(x)
        return torch.tanh(mean)


if __name__ == "__main__":

    args = tyro.cli(Args)
    time_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
    run_name = f"{args.env_id}__{args.exp_name}__seed-{args.seed}__norm-{args.norm_obs}__{time_str}"
    save_dir = f"./gym_results/{args.env_id}__{args.exp_name}__norm-{args.norm_obs and args.norm_reward}__ln-{args.use_ln}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # steps done before training (used for training resume)
    init_step = -1 

    # env for eval
    env_ = gym.make(args.env_id)
    env_ = gym.wrappers.FlattenObservation(env_)
    eval_env = gym.wrappers.RecordEpisodeStatistics(env_)
    eval_env.reset(seed=args.seed+100)

    # obs/reward norm
    obs_stats = SampleMeanStd(shape=envs.single_observation_space.shape, device=device)
    reward_stats = SampleStd(shape=(), device=device)
    reward_trace = np.zeros(args.num_envs)

    max_action = float(envs.single_action_space.high[0])

    hidden_size = args.hidden_size
    actor = Actor(envs, use_ln=args.use_ln, hidden_size=hidden_size)
    qf1 = SoftQNetwork(envs, use_ln=args.use_ln, hidden_size=hidden_size)
    qf2 = SoftQNetwork(envs, use_ln=args.use_ln, hidden_size=hidden_size)

    qf1_target = SoftQNetwork(envs, use_ln=args.use_ln, hidden_size=hidden_size).to(device)
    qf2_target = SoftQNetwork(envs, use_ln=args.use_ln, hidden_size=hidden_size).to(device)
    target_actor = Actor(envs, use_ln=args.use_ln, hidden_size=hidden_size).to(device)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        #log_alpha = torch.ones(1, requires_grad=True, device=device) * (-1.5)
        log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
        a_optimizer = None

    # load checkpoint if provided
    if len(args.resume_training) > 0:

        # create opts before loading checkpoint
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

        log_alpha_ = log_alpha if args.autotune else 0.0 # if no autotune we don't need to restore alpha

        # load checkpoint (and get initial step)
        init_step = load_sac_checkpoint(
            path=args.resume_training,
            actor=actor,
            qf1=qf1,
            qf2=qf2,
            actor_opt=actor_optimizer,
            q_opt=q_optimizer,
            device=device,
            obs_stats=obs_stats,
            reward_stats=reward_stats,
            log_alpha=log_alpha_,
            alpha_opt=a_optimizer
        )

        if args.autotune:
            alpha = log_alpha.exp().item()

        print(f"Checkpoint loaded: step {init_step}")
        print("------------------------------------------------------------------------------")

    if init_step == -1:
        # if init_step still = -1 -> no checkpoint used. We can move to device and init opts

        # move to device
        actor.to(device)
        qf1.to(device)
        qf2.to(device)

        # init opts
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())



    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    
    # Restore old logs if possible
    if init_step > -1:
        old_logs_path = os.path.join(save_dir, f"seed_{args.seed}.npz")
        returns, term_time_steps, eval_returns =  restore_logs(old_logs_path, init_step)
    else:
        returns, term_time_steps, eval_returns = [], [], []  # training log

    if args.norm_obs:
        obs_stats.update(obs)

    for global_step in range(init_step + 1, args.total_timesteps):

        """####### EVAL + CHECKPOINT SAVE #######"""

        if args.eval_freq > 0 and ((global_step % args.eval_freq == 0) or (global_step == (args.total_timesteps-1))) and global_step > 0:
            #eval
            stats = obs_stats if args.norm_obs else None
            eval_avg_return = evaluate_sac(actor, eval_env, global_step, device, args.seed+100, stats, deterministic=args.deterministic_eval)
            eval_returns.append(eval_avg_return)

            np.savez(
                os.path.join(save_dir, "seed_{}.npz".format(args.seed)),
                eval_returns = np.array(eval_returns),
                training_returns = np.array(returns),
                term_time_steps = term_time_steps
                )
        
            if args.checkpoint_freq > 0 and (global_step % args.checkpoint_freq == 0):
                # save checkpoint (models + opt + stats)
                save_sac_checkpoint(
                    path=save_dir,
                    actor=actor,
                    qf1=qf1,
                    qf2=qf2,
                    actor_opt=actor_optimizer,
                    q_opt=q_optimizer,
                    seed=args.seed,
                    global_step=global_step,
                    log_alpha=math.log(alpha) if not args.autotune else log_alpha,
                    alpha_opt=a_optimizer,
                    obs_stats=obs_stats,
                    reward_stats=reward_stats
                )

                # save json with step, avg eval return and name checkpoint
                log_checkpoint_stats(
                    dir_path=save_dir,
                    steps_done=global_step,
                    avg_return=eval_avg_return,
                    seed=args.seed
                    )

                print(f"--------- checkpoint {global_step} saved ----------")
                
        """####### END - EVAL + CHECKPOINT SAVE #######"""
            
            
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_to_agent = obs_stats.normalize(obs) if args.norm_obs else torch.Tensor(obs).to(device)
            actions, _, _ = actor.get_action(obs_to_agent)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if args.norm_obs:
            obs_stats.update(next_obs)
        if args.norm_reward:
            reward_trace = reward_trace * (1 - (terminations or truncations)) * args.gamma + rewards
            reward_stats.update(reward_trace)

        alpha_ = alpha / float((np.sqrt(reward_stats.var+1e-08))) if args.norm_reward else alpha

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    returns.append(info['episode']['r'][0])
                    term_time_steps.append(global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        steps_since_resume = global_step if init_step == -1 else global_step - init_step - 1
        if steps_since_resume > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():

                next_observations = obs_stats.normalize(data.next_observations) if args.norm_obs else data.next_observations

                next_state_actions, next_state_log_pi, _ = actor.get_action(next_observations)
                qf1_next_target = qf1_target(next_observations, next_state_actions)
                qf2_next_target = qf2_target(next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha_ * next_state_log_pi

                data_rewards = reward_stats.normalize(data.rewards).flatten() if args.norm_reward else data.rewards.flatten()

                next_q_value = data_rewards + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            observations = obs_stats.normalize(data.observations) if args.norm_obs else data.observations

            qf1_a_values = qf1(observations, data.actions).view(-1)
            qf2_a_values = qf2(observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(observations)
                    qf1_pi = qf1(observations, pi)
                    qf2_pi = qf2(observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha_ * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    if args.end_checkpoint or (args.checkpoint_freq > 0 and args.total_timesteps % args.checkpoint_freq == 0):

        save_sac_checkpoint(
                path=save_dir,
                actor=actor,
                qf1=qf1,
                qf2=qf2,
                actor_opt=actor_optimizer,
                q_opt=q_optimizer,
                seed=args.seed,
                global_step=args.total_timesteps,
                log_alpha=math.log(alpha) if not args.autotune else log_alpha,
                alpha_opt=a_optimizer,
                obs_stats=obs_stats,
                reward_stats=reward_stats
                            )

        log_checkpoint_stats(
                    dir_path=save_dir,
                    steps_done=args.total_timesteps,
                    avg_return=eval_avg_return,
                    seed=args.seed
                )
        

    envs.close()
