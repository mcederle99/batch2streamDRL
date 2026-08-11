"""Code modified from CleanRL repo: https://github.com/vwxyzjn/cleanrl/tree/master"""
import os
import random
import time
from dataclasses import dataclass
import datetime

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
from eval_utils import evaluate_td3
from checkpoints_utils import save_td3_checkpoint, log_checkpoint_stats, load_td3_checkpoint, restore_logs 

from sparse_init import sparse_init
from optim import NormSGD

def compute_weight_norm(model):
    """
    Compute the L2 (Frobenius) norm of the effective weights of the model.
    Correctly handles Linear layers with and without weight normalization.
    """
    total_norm_sq = 0.0

    for module in model.modules():
        # Handle Linear layers explicitly
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
                # Weight-normalized Linear layer
                v = module.weight_v
                g = module.weight_g

                # Effective weight: g * v / ||v||
                v_norm = torch.norm(v, p=2)
                effective_weight = g * v / v_norm

                total_norm_sq += torch.norm(effective_weight, p=2).pow(2).item()
            else:
                # Standard Linear layer
                total_norm_sq += torch.norm(module.weight, p=2).pow(2).item()

            # Optional: include bias
            if module.bias is not None and module.bias.requires_grad:
                total_norm_sq += torch.norm(module.bias, p=2).pow(2).item()

        # Handle all other parameters (e.g. Conv, BatchNorm, etc.)
        else:
            for param in module.parameters(recurse=False):
                if param.requires_grad:
                    total_norm_sq += torch.norm(param, p=2).pow(2).item()

    return total_norm_sq ** 0.5

def compute_gradient_norm(model):
    """
    Compute the norm of all gradients in the model.
    Useful for gradient clipping and monitoring training.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

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
    """the id of the environment"""
    total_timesteps: int = 3000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
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
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    norm_obs: bool = False
    """normalize obs"""
    norm_reward: bool = False
    """normalize reward"""
    use_ln : bool = False
    """use layer norm"""
    hidden_size : int = 128
    """hidden size MLPs"""
    opt : str = "adam"

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
class QNetwork(nn.Module):
    def __init__(self, env, use_ln=False, hidden_size=128):
        super().__init__()
        self.fc_layer = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            hidden_size,
        )
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, 1)
        self.use_ln = use_ln

        self.apply(initialize_weights)

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


class Actor(nn.Module):
    def __init__(self, env, use_ln=False, hidden_size=128):
        super().__init__()
        self.fc_layer = nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, np.prod(env.single_action_space.shape))
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

        self.apply(initialize_weights)

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

        x = torch.tanh(self.linear_mu(x))

        if self.use_ln and squeeze:
            x = x.squeeze(dim=0)

        return x * self.action_scale + self.action_bias


if __name__ == "__main__":

    args = tyro.cli(Args)
    time_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
    run_name = f"{args.env_id}__{args.exp_name}__seed-{args.seed}__norm-{args.norm_obs}__{time_str}"
    save_dir = f"./gym_results/{args.env_id}__{args.exp_name}__norm-{args.norm_obs and args.norm_reward}__ln-{args.use_ln}__batch-{args.batch_size}__opt-{args.opt}"
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

    hidden_size = args.hidden_size
    actor = Actor(envs, use_ln=args.use_ln, hidden_size=hidden_size)
    qf1 = QNetwork(envs, use_ln=args.use_ln, hidden_size=hidden_size)
    qf2 = QNetwork(envs, use_ln=args.use_ln, hidden_size=hidden_size)

    qf1_target = QNetwork(envs, use_ln=args.use_ln, hidden_size=hidden_size).to(device)
    qf2_target = QNetwork(envs, use_ln=args.use_ln, hidden_size=hidden_size).to(device)
    target_actor = Actor(envs, use_ln=args.use_ln, hidden_size=hidden_size).to(device)

    # load checkpoint if provided
    if len(args.resume_training) > 0:

        # create opts before loading checkpoint
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

        # load checkpoint (and get initial step)
        init_step = load_td3_checkpoint(
            path=args.resume_training,
            actor=actor,
            qf1=qf1,
            qf2=qf2,
            actor_opt=actor_optimizer,
            q_opt=q_optimizer,
            device=device,
            obs_stats=obs_stats,
            reward_stats=reward_stats
        )

        print(f"Checkpoint loaded: step {init_step}")
        print("------------------------------------------------------------------------------")

    if init_step == -1:

        # move to device
        actor.to(device)
        qf1.to(device)
        qf2.to(device)

        # init opts
        if args.opt == "adam":
            q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
            actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
        elif args.opt == "nsgd":
            print("nsgd used!!")
            q_optimizer = NormSGD(list(qf1.parameters()) + list(qf2.parameters()), lr=0.5)
            actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
        else:
            raise NotImplemented



    target_actor.load_state_dict(actor.state_dict())
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
    if args.norm_obs:
        obs_stats.update(obs)

    # Restore old logs if possible
    if init_step > -1:
        old_logs_path = os.path.join(save_dir, f"seed_{args.seed}.npz")
        returns, term_time_steps, eval_returns =  restore_logs(old_logs_path, init_step)
    else:
        returns, term_time_steps, eval_returns = [], [], []  # training log

    grads_q, grads_p = [], []
    norm_q, norm_p = [], []

    norm_p.append(compute_weight_norm(actor))
    norm_q.append(compute_weight_norm(qf1))

    for global_step in range(init_step + 1, args.total_timesteps):

        """####### EVAL + CHECKPOINT SAVE #######"""

        if args.eval_freq > 0 and ((global_step % args.eval_freq == 0) or (global_step == (args.total_timesteps-1))) and global_step > 0:
            #eval
            stats = obs_stats if args.norm_obs else None
            eval_avg_return = evaluate_td3(actor, eval_env, global_step, device, args.seed+100, stats)
            eval_returns.append(eval_avg_return)

            np.savez(
                os.path.join(save_dir, "seed_{}.npz".format(args.seed)),
                eval_returns = np.array(eval_returns),
                training_returns = np.array(returns),
                term_time_steps = term_time_steps,
                    norm_q = np.array(norm_q),
                    grad_q=np.array(grads_q),
                    norm_p = np.array(norm_p),
                    grad_p= np.array(grads_p)
                )

            if args.checkpoint_freq > 0 and (global_step % args.checkpoint_freq == 0):
                # save checkpoint (models + opt + stats)
                save_td3_checkpoint(save_dir,
                                    actor=actor,
                                    qf1=qf1,
                                    qf2=qf2,
                                    actor_opt=actor_optimizer,
                                    q_opt=q_optimizer,
                                    global_step=global_step,
                                    seed=args.seed,
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
            with torch.no_grad():
                obs_to_agent = obs_stats.normalize(obs) if args.norm_obs else torch.Tensor(obs).to(device)
                actions = actor(obs_to_agent)
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if args.norm_obs:
            obs_stats.update(next_obs)
        if args.norm_reward:
            reward_trace = reward_trace * (1 - (terminations or truncations)) * args.gamma + rewards
            reward_stats.update(reward_trace)

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
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_observations = obs_stats.normalize(data.next_observations) if args.norm_obs else data.next_observations

                next_state_actions = (target_actor(next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(next_observations, next_state_actions)
                qf2_next_target = qf2_target(next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)

                data_rewards = reward_stats.normalize(data.rewards).flatten() if args.norm_reward else data.rewards.flatten()

                next_q_value = data_rewards + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            observations = obs_stats.normalize(data.observations) if args.norm_obs else data.observations

            qf1_a_values = qf1(observations, data.actions).view(-1)
            qf2_a_values = qf2(observations, data.actions).view(-1)
            if args.opt == "adam":
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss
            elif args.opt == "nsgd":
                delta_1 = (next_q_value - qf1_a_values).detach()
                delta_2 = (next_q_value - qf2_a_values).detach()
                scale_1 = torch.clamp(delta_1, min=-1.0, max=1.0)
                scale_2 = torch.clamp(delta_2, min=-1.0, max=1.0)

                qf1_loss = -qf1_a_values * scale_1
                qf2_loss = -qf2_a_values * scale_2
                qf_loss = torch.mean(qf1_loss + qf2_loss)
            else:
                raise NotImplemented

            

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            grads_q.append(compute_gradient_norm(qf1))
            q_optimizer.step()
            norm_q.append(compute_weight_norm(qf1))

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(observations, actor(observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                grads_p.append(compute_gradient_norm(actor))
                actor_optimizer.step()
                norm_p.append(compute_weight_norm(actor))

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    if args.end_checkpoint or (args.checkpoint_freq > 0 and args.total_timesteps % args.checkpoint_freq == 0):
        save_td3_checkpoint(path = save_dir,
                            seed=args.seed,
                            actor=actor,
                            qf1=qf1,
                            qf2=qf2,
                            actor_opt=actor_optimizer,
                            q_opt=q_optimizer,
                            global_step=args.total_timesteps,
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
