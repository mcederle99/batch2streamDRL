"""Checkpoint utils: save checkpoint for SAC and TD3"""


import json
from pathlib import Path
import numpy as np
import os

import torch
from torch import nn
from torch.optim import Optimizer

from normalization_utils import SampleMeanStd, SampleStd

def format_steps(steps: int) -> str:
    if steps >= 1_000_000_000:
        return f"{steps / 1_000_000_000:.2f}G"
    elif steps >= 1_000_000:
        return f"{steps / 1_000_000:.2f}M"
    elif steps >= 1_000:
        return f"{steps / 1_000:.2f}K"
    else:
        return str(steps)
    


def log_checkpoint_stats(
    dir_path: str,
    steps_done: int,
    avg_return: float,
    seed : int
):
    """
    Appends an evaluation result to a JSONL file (used for each checkpoint).

    Each line has the form:
    {"steps": int, "avg_return": float, "tag" : str}
    tag = name file checkpoint
    """


    step_tag = format_steps(steps_done)
    tag_checkpoint = f"checkpoint_{seed}_{step_tag}.pt"

    record = {
        "steps": steps_done,
        "avg_return": avg_return,
        "tag" : tag_checkpoint
    }

    file_path = dir_path + f"/checkpoints_seed-{seed}.jsonl"
    path = Path(file_path)

    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")



def restore_logs(npz_path, init_step):
    if init_step > -1 and os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        return (data["eval_returns"].tolist()[:init_step],
                data["training_returns"].tolist()[:init_step],
                data["term_time_steps"].tolist()[:init_step])
    else:
        return [], [], []


""" -------------------------- TD3 ----------------------------------"""


def save_td3_checkpoint(
        path : str,
        seed : int,
        actor : nn.Module,
        qf1 : nn.Module,
        qf2 : nn.Module,
        actor_opt : Optimizer,
        q_opt : Optimizer,
        global_step : int,
        obs_stats: SampleMeanStd = None,
        reward_stats: SampleStd = None,
):
    step_tag = format_steps(global_step)

    checkpoint = {
        "actor" : actor.state_dict(),
        "qf1" : qf1.state_dict(),
        "qf2" : qf2.state_dict(),
        "actor_opt" : actor_opt.state_dict(),
        "q_opt" : q_opt.state_dict(),
        "global_step" : global_step

    }

    # Save stats if provided
    if obs_stats is not None:
        checkpoint["obs_stats"] = {
            "mean": obs_stats.mean,
            "var": obs_stats.var,
            "p": obs_stats.p,
            "count": obs_stats.count
        }

    if reward_stats is not None:
        checkpoint["reward_stats"] = {
            "mean": reward_stats.mean,
            "var": reward_stats.var,
            "p": reward_stats.p,
            "count": reward_stats.count
        }

    file_path = path + f"/checkpoint_{seed}_{step_tag}.pt"

    torch.save(checkpoint, file_path)
    return file_path 

def load_td3_checkpoint(
        path : str,
        actor : nn.Module,
        qf1 : nn.Module,
        qf2 : nn.Module,
        actor_opt : Optimizer,
        q_opt : Optimizer,
        device,
        obs_stats: SampleMeanStd = None,
        reward_stats: SampleStd = None,       
):
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except:
        print("Not valid path!!!!!!!")
        print("------------------- Training from scratch ---------------------------------")
        return -1

    # load on cpu
    actor.load_state_dict(checkpoint["actor"])
    qf1.load_state_dict(checkpoint["qf1"])
    qf2.load_state_dict(checkpoint["qf2"])

    # load opt (on cpu)
    actor_opt.load_state_dict(checkpoint["actor_opt"])
    q_opt.load_state_dict(checkpoint["q_opt"])

    # move to device
    actor.to(device)
    qf1.to(device)
    qf2.to(device)

    # move opt to device
    for optimizer in [actor_opt, q_opt]:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # Restore stats
    if obs_stats is not None and "obs_stats" in checkpoint:
        s = checkpoint["obs_stats"]
        obs_stats.mean = s["mean"]
        obs_stats.var = s["var"]
        obs_stats.p = s["p"]
        obs_stats.count = s["count"]

    if reward_stats is not None and "reward_stats" in checkpoint:
        s = checkpoint["reward_stats"]
        reward_stats.mean = s["mean"]
        reward_stats.var = s["var"]
        reward_stats.p = s["p"]
        reward_stats.count = s["count"]

    return checkpoint["global_step"]




"""-------------------------------- SAC -------------------------------------"""


def save_sac_checkpoint(
        path : str,
        seed : int,
        actor : nn.Module,
        qf1 : nn.Module,
        qf2 : nn.Module,
        actor_opt : Optimizer,
        q_opt : Optimizer,
        log_alpha, 
        global_step : int,
        obs_stats: SampleMeanStd = None,
        reward_stats: SampleStd = None,
        alpha_opt : Optimizer = None
):
    step_tag = format_steps(global_step)

    checkpoint = {
        "actor" : actor.state_dict(),
        "qf1" : qf1.state_dict(),
        "qf2" : qf2.state_dict(),
        "actor_opt" : actor_opt.state_dict(),
        "q_opt" : q_opt.state_dict(),
        "global_step" : global_step

    }

    # Save stats if provided
    if obs_stats is not None:
        checkpoint["obs_stats"] = {
            "mean": obs_stats.mean,
            "var": obs_stats.var,
            "p": obs_stats.p,
            "count": obs_stats.count
        }

    if reward_stats is not None:
        checkpoint["reward_stats"] = {
            "mean": reward_stats.mean,
            "var": reward_stats.var,
            "p": reward_stats.p,
            "count": reward_stats.count
        }

    # alpha
    log_alpha_ = log_alpha.detach().cpu().numpy().item() if isinstance(log_alpha, torch.Tensor) else log_alpha
    checkpoint["log_alpha"] = log_alpha_

    if alpha_opt is not None:
        checkpoint["alpha_opt"] = alpha_opt.state_dict()


    file_path = path + f"/checkpoint_{seed}_{step_tag}.pt"

    torch.save(checkpoint, file_path)
    return file_path 


def load_sac_checkpoint(
        path : str,
        actor : nn.Module,
        qf1 : nn.Module,
        qf2 : nn.Module,
        actor_opt : Optimizer,
        q_opt : Optimizer,
        log_alpha,
        device,
        obs_stats: SampleMeanStd = None,
        reward_stats: SampleStd = None,
        alpha_opt: Optimizer = None       
):
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except:
        print("Not valid path!!!!!!!")
        print("------------------- Training from scratch ---------------------------------")
        return -1

    # load on cpu
    actor.load_state_dict(checkpoint["actor"])
    qf1.load_state_dict(checkpoint["qf1"])
    qf2.load_state_dict(checkpoint["qf2"])

    # load opt (on cpu)
    actor_opt.load_state_dict(checkpoint["actor_opt"])
    q_opt.load_state_dict(checkpoint["q_opt"])

    # move to device
    actor.to(device)
    qf1.to(device)
    qf2.to(device)

    # move opt to device
    for optimizer in [actor_opt, q_opt]:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # Restore stats
    if obs_stats is not None and "obs_stats" in checkpoint:
        s = checkpoint["obs_stats"]
        obs_stats.mean = s["mean"]
        obs_stats.var = s["var"]
        obs_stats.p = s["p"]
        obs_stats.count = s["count"]

    if reward_stats is not None and "reward_stats" in checkpoint:
        s = checkpoint["reward_stats"]
        reward_stats.mean = s["mean"]
        reward_stats.var = s["var"]
        reward_stats.p = s["p"]
        reward_stats.count = s["count"]

    # alpha

    if isinstance(log_alpha, torch.Tensor):
        with torch.no_grad():
            log_alpha.fill_(float(checkpoint["log_alpha"]))
    else:
        log_alpha = float(checkpoint["log_alpha"])

    if alpha_opt is not None:
        alpha_opt.load_state_dict(checkpoint["alpha_opt"])

        for state in alpha_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)          
    

    return checkpoint["global_step"]