# batch2streamDRL

## Create the conda environment

We suggest creating a conda environment using the provided .yml file:

```
conda env create -f environment_mac.yml
```

Replace `mac` with `linux` if you are using a Linux machine. 

## Content per directory

* *batch_methods*: code for TD3 and SAC e.g. original TD3 and SAC, + normalization, evaluation code, TD3+SGDC etc.
* *streaming_methods*: code for Stream AC, SDAC, S2AC and finetuning.
* *xmls_modified*: .xml for dog-walk-v0-p, walker-run-v0-p1, and walker-run-v0-p2.

## Run streaming experiments

Enter into  *streaming_methods* directory:  ```cd streaming_methods```. To execute one the three algorithms:

* **Stream AC**: ```python python stream_ac_continuous.py --env_name Hopper-v4 --debug```
* **SDAC (Ours)**: ```python stream_dac_continuous.py --no_time_info --debug --target_noise y --optim adam --env_name Hopper-v4```
* **S2AC (Ours)**: ```python stream_sac_continuous.py --squash --no_time_info --optim adam --eta_decay_type reward_based --env_name Hopper-v4```

Assuming ```shimmy``` is installed, you can execute DM Control Suite experiments by specifying ```--env_name dm_control/domain-task-v0```. Example: ```--env_name dm_control/walker-run-v0```.

All results are saved in a subdirectory *stream_results*. For each (algorithm, env) combination a directory is created e.g. streamdac-Hopper-v4_+,  streamdac-dm_control/quadruped-run-v0_+ (+ = addional parameters). In each directory a file *evalreturns_i.npy* , where i = seed, is created at the end of the run.

**Example**: SDAC with 0.0003 actor learning rate, adam as actor optimizer, with target noise (y = use noise target), 2 envs (Hopper-v4, dm_control/walker-run-v0)

```
code_root/
└── streaming_methods/
    └── stream_results/
        └──  streamdac-Hopper-v4_0.0003_adam_y/
        	├── evalreturns_1.npy
        	└── evalreturns_2.npy
        └──streamdac-dm_control/
        	└── walker-run-v0_0.0003_adam_y
        	  ├── evalreturns_1.npy
        	  └── evalreturns_2.npy
```

### Reproduce ablation

To disable the use of the target noise in SDAC just set ```--target_noise n```, while to not scale alpha in S2AC set ```--eta_decay_type none```

## Run batch experiments

Example for **TD3+norm** on quadruped-run-v0:

```bash
python td3_continuous_action_norm.py --seed 0 --checkpoint_freq 100000 --norm_obs --norm_reward --use_ln --env_id dm_control/quadruped-run-v0
```

Example for **TD3** (no normalization, scaled reward, layer norm) on Hopper-v4:

```bash
python td3_continuous_action_norm.py --seed 0 --checkpoint_freq 100000 --env_id Hopper-v4
```

Same for **SAC+norm**:

```bash
python sac_continuous_action_norm.py --seed 0 --checkpoint_freq 100000 --norm_obs --norm_reward --use_ln --env_id dm_control/quadruped-run-v0
```

and for **SAC**:

```bash
python sac_continuous_action_norm.py --seed 0 --checkpoint_freq 100000 --env_id dm_control/quadruped-run-v0
```

## Run TD3+SGDC

nsgd = sgdc for the critic

```bash
python td3_continuous_action_norm-mix.py --seed 0 --checkpoint_freq 100000 --norm_obs --norm_reward --use_ln --env_id dm_control/quadruped-run-v0 --opt nsgd
```

## Checkpoints and results for batch methods

For each batch algorithm and environment, all logged data are saved in a specific directory. For instance, if *td3_continuous_action_norm-mix.py* (TD3 norm with the addional arg for the use of SGDC) is executed for environment *dm_control/walker-run-v0* all the results are saved in ```code_root/batch_methods/gym_results/dm_control/walker-run-v0__td3_continuous_action_norm-mix__norm-True__ln-True__batch-256__opt-nsgd```, where *norm-True* denotes the use of state normalization and scaled reward (e.g. TD3-norm), *batch-256* denotes the use of 256 as batch size and *opt-nsgd* denotes the use of SGDC as optimizer for the critic. 

### Eval results

In the directory for the couple (batch algorithm, env), a *seed_i.npz* file is created for each training seed *i*. This file is periodically updated during training, namely, it is not saved at the end of training. By ```np.load(path)["eval_returns"]``` (path = path to *seed_i.npz*), the numpy array with the average return over 10 evaluation episodes at each evaluation step (e.g. every 10_000 steps) is returned.

### Checkpoints

The argument ```--checkpoint_freq``` is used to specify the amount of steps between checkpoints. During training two files are created per checkpoint: *checkpoint_i_steps.pt* and *checkpoints-seed-i.jsonl*. *checkpoint_i_steps.pt* (for instance *checkpoint_0_1.50M.pt* for seed 0 and checkpoint at 1.5 * 10^6 steps) is a torch checkpoint with several keys:
1. actor: actor state dict
2. qf1: first q net state dict
3. qf2: second q net state dict
4. actor_opt: actor opt state dict
5. q_opt: q nets opt state dict
6. global_step
7. obs_stats: with several (key, value) for state stats. keys: mean, var, p, count
8. reward_stats: same as obs stats
9. (for SAC): alpha

*checkpoints-seed-i.jsonl* stores a json object per checkpoint for seed i. Each json object has three keys: 
1. steps: number of steps for the checkpoint
2. tag: name checkpoint e.g. *checkpoint_0_1.50M.pt*
3. avg_return: average return over 10 episodes for the agent saved in that checkpoint.

This last file is useful to have access to the performance of the agent saved at each checkpoint.

## Batch-to-Stream RL from TD3 to SDAC

If you want to use the modified environments for ```dog-walk-v0``` and ```walker-run-v0``` you need first of all to copy the modified .xml files present in the folder ```xmls_modified/``` into the correct directory, using a command similar to this:

```bash
cp xmls_modified/dog_modified.xml /opt/anaconda3/envs/mujoco/lib/python3.8/site-packages/dm_control/suite/dog.xml
# or
cp xmls_modified/walker_modified.xml /opt/anaconda3/envs/mujoco/lib/python3.8/site-packages/dm_control/suite/walker.xml
```

After that, you need to choose the pre-training checkpoint that you want to use, e.g. you may choose the checkpoint at 1.50M steps of ```dog-walk-v0``` using TD3 with SGDC (pre-training seed 0). Finally, you can run the script ```finetuning_td3.py``` using the following command line arguments:

```bash
python finetuning_td3.py \
--seed 0 \
--env_id dm_control/dog-walk-v0 \
--debug \
--starting_checkpoint 1.50M \
--pre_training_seed 1 \
--checkpoint_dir finetuning_checkpoints/checkpoints_dog-walk-v0-p/checkpoint_1_1.50M.pt \
--mixed _mixed 
```

The results will be saved in the following path:

```
code_root/
└── streaming_methods/
    └── finetuning_results/
		└── dm_control/
			└──  dog-walk-v0_streamdac_2_1.50M_0_adam_5000_0.1_0.1_mixed_256.0/
				└── evalreturns_0.npy
```

where '2' is the value of $\kappa_q$, 'adam' is the actor optimizer chosen for finetuning, '5000' is the number of warmup steps, '0.1' and '0.1' are the standard deviations of the exploration and target noise respectively, and '256.0' is the value by which we divide the actor learning rate when moving from pre-training to finetuning. All these values can be changed at will to test different configurations.

## Finetuning checkpoints
The `finetuning_checkpoints/` directory contains one subdirectory per finetuning environment, each holding pre-trained models with TD3-norm-SGDC that can be used to run finetuning experiments with SDAC. Obviously, for perturbated environments the pre-training stage has been performed with the original environment.