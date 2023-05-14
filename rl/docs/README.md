# Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training - Code](#training---code)
  - [Training - Configuration](#training---configuration)
  - [Training - Execution](#training---execution)
  - [Policy Evaluation](#policy-evaluation)
- [Environment Parameters Curriculum](#environment-parameters-curriculum)
- [SLURM Support](#slurm-support)

# Installation
- Note: The training code does not work on MacOS because of the reverb dependency. Policy evaluation works on MacOS.
- Install [daemontools](https://cr.yp.to/daemontools.html):
  - `sudo apt install daemontools` if you have admin access
  - Follow instructions at
  [samarth-robo/daemontools-0.76](https://github.com/samarth-robo/daemontools-0.76/tree/local_install) otherwise.
  Note: Use the `local_install` branch and not `master`.
- `(base) $ conda env create -f environment.yml`. Remove `[reverb]` from `tf-agents[reverb]==0.10.0` if installing on MacOS. Hence training will not work on MacOS, only policy execution will.
- `(base) $ conda activate sac_utils`

# Usage
This repository provides the SAC RL training framework, you need to provide the RL environment.

## Training - Code
- Create your RL environment, deriving from
[TensorFlow Agents `PyEnvironment`](https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/PyEnvironment).
Alternatively, you can load pre-made environments from the
[Gym suite](https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/suite_gym/load).
- Note, this environment should override `PyEnvironment`'s
[`set_state()`](https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/PyEnvironment#set_state), where
the `state` argument will be a dictionary. See how `self.collect_env.set_state()` is used in
[`tfagents_system/workers.py`](../tfagents_system/workers.py).
- The `TrainParamsBase` abstract base class in
[`tfagents_system/train_params_base.py`](../tfagents_system/train_params_base.py) collects the SAC training hyperparameters
and the [SacAgent](https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/SacAgent) building code.
- Derive your own class from `TrainParamsBase` and fill out the `env_ctor()` function. This class _must_ be named
`TrainParams`.
- `env_ctor()` can use the dictionary `TrainParamsBase.env_config`, which is created from the configuration JSON
(see the [Training - Configuration](#training---configuration) section below for details). 
- Optionally, override the `collect_env_args_kwargs()` and `eval_env_args_kwargs()` functions, which return the `args`
tuple and `kwargs` dictionary that will be used while building the experience collection and evaluation environments
respectively. See how `collect_env` and `eval_env` are created in
[`tfagents_system/workers.py`](../tfagents_system/workers.py) using those functions. They can also use the `env_config`
dictionary.
- For an example of the
[Minitaur environment](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py),
see [`tfagents_system/minitaur_params.py`](../tfagents_system/minitaur_params.py) and
[`config/minitaur.json`](../config/minitaur.json).

## Training - Configuration
Training involves `N+2` separate programs, where `N` is the number of parallel workers.
- policy update program, created by [`tfagents_system/sac_train.py`](../tfagents_system/sac_train.py)
- replay buffer program, created by [`tfagents_system/sac_replay_buffer.py`](../tfagents_system/sac_replay_buffer.py)
- `N` x agents with their own experience collection and evaluation environments, created by
[`tfagents_system/sac_collect.py`](../tfagents_system/sac_collect.py) and
[`tfagents_system/workers.py`](../tfagents_system/workers.py).

CPU allocation to these programs is controlled by environment variables in
[`training_variables.sh`](../training_variables.sh).
| Task | #CPUs |
| --- | --- |
| Reverb replay buffer | `SAC_REVERB_CPUS` |
| Policy update | `SAC_TRAIN_CPUS` |
| Parallel experience collection and evaluation workers | `SAC_NUM_COLLECT_WORKERS * SAC_COLLECT_WORKER_CPUS` |

In [`training_variables.sh`](../training_variables.sh):
- Set `SAC_PARAMS` to the Python file containing the class that derives from `TrainParamsBase`.
- Set `SAC_CONFIG` to the JSON file containing environment parameters, available as the dictionary
`TrainParamsBase.env_config`.

For an example of the
[Minitaur environment](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py),
see [`training_variables_minitaur.sh`](../training_variables_minitaur.sh).

## Training - Execution
```bash
$ ./run_script.sh <training variables .sh file>
```

For example, to train the
[Minitaur environment](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py):
`./run_script.sh training_variables_minitaur.sh`.

## Policy Evaluation
```bash
(sac_utils) $ python tfagents_system/run.py \
--params <Python file instantiating TrainParamsBase e.g. $SAC_PARAMS> \
--exp_dir <Path to experiment directory relative to repository root e.g. logs/${SAC_EXP_NAME}> \
--config <JSON environment config e.g. $SAC_CONFIG> \
--num_episodes <int number of episodes the policy will be evaluated for> \
--checkpoint <name of the checkpoint directory within exp_dir/policies/checkpoints> \
--seed <int seed that will be passed to the environment constructor>
```

For example, running

```bash
(sac_utils) $ python tfagents_system/run.py --params tfagents_system/minitaur_params.py --exp_dir logs/minitaur \
--config config/minitaur.json --num_episodes 5 --checkpoint policy_checkpoint_0000500000 --seed 108
```

produces a video like the following:

https://user-images.githubusercontent.com/2848070/205775251-dc1909da-d73e-4b4f-bc4a-95c8265df648.mp4

# Environment Parameters Curriculum
Sometimes it is useful to vary environment parameters in a curriculum that depends on the iteration number. An
"iteration" means one cycle of
- collecting experience with all workers,
- pushing all workers' experience to the common replay buffer,
- sampling a configured number of experience minibatches from the replay buffer,
- updating the policy with the gradients of the SAC loss function, and
- pushing the updated policy to all the workers.

For example, if the parameters contain the range of values from which environment properties will be randomly sampled,
we might want to start with a small range and widen it as the training progresses and the policy improves. This is 
called _domain randomization_, and often the robot agent's physics parameters like friction, mass, etc. are randomized.
This is supported here. Example section of the JSON:

```json
"curriculum": {
  "mass": {
    "iters": [0, 20000, 30000, 40000, 50000],
    "range": [0.0, 10.0]
  },
  "friction": {
    "iters": [50000, 60000, 70000, 90000, 120000],
    "range": [0.1, 0.5]
  }
}
```

For each parameter like mass and friction, `range = [range_min, range_max]` is divided unformly linearly using
`values = np.linspace(range_min, range_max, len(iters))`. So the `values` array is the same size as `iters`. The
following pseudo-code is applied by each parallel worker to each parameter of its environment (shown for the mass
parameter only):

```python
for iteration_idx in range(num_iterations):
  while len(iters) and (iteration_idx >= iters[0]):
    iters.popleft()
    value = values.popleft()
    env.set_state({"mass": value})
  # continue with the iteration i.e. collect experience in this configured environment
```

In the example above, the friction parameter curriculum will not start till iteration 50000. So the value of friction
before that will be the one used by the constructor. This can be the default value hardcoded in the environment code,
or `TrainParamsBase.collect_env_args_kwargs()` and `TrainParamsBase.eval_env_args_kwargs()` can include the key-value
pair `friction: <value>` in the `kwargs` dictionary they return.

# SLURM Support
When training on a compute cluster managed by SLURM, allot
`SAC_REVERB_CPUS + SAC_TRAIN_CPUS + SAC_NUM_COLLECT_WORKERS * SAC_COLLECT_WORKER_CPUS + 2` CPUs.
[`train_script.sh`](../train_script.sh), which is called by [`run_script.sh`](../run_script.sh), uses
`cat /proc/self/status` to get the CPU IDs available to this SLURM job. It then uses `taskset` to allocate the
appropriate IDs of CPUs to each program. The `SLURM_PROCID` environment variable is also set for each worker program.
