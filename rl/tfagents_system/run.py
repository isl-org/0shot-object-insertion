import init_paths
import argparse
import imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from utilities import setup_logging, import_params_module

from tf_agents.drivers.py_driver import PyDriver
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.policies import policy_loader
from tfagents_system.train_params_base import TrainParamsBase


osp = os.path


if __name__ == "__main__":
    # cli
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--params", type=str, default="tfagents_system/minitaur_params.py", help="params file")
    parser.add_argument("-e", "--exp_dir", help="experiment dir", required=True, type=str)
    parser.add_argument("-c", "--config", type=str, default="config/minitaur.json", help="config file")
    parser.add_argument("-n", "--num_episodes", help="number of episodes", default=25, type=int)
    parser.add_argument("--checkpoint", required=True,
                        help="path to ckpt dir in exp_dir/policies/checkpoints e.g. policy_checkpoint_0000500000")
    parser.add_argument('--seed', type=int, default=108)
    args = parser.parse_args()

    setup_logging()

    # load policy
    policy_dir = osp.join(args.exp_dir, 'policies', 'greedy_policy')
    ckpt_dir = osp.join(args.exp_dir, 'policies', 'checkpoints', args.checkpoint)
    print(f'Loading policy from {policy_dir} and {ckpt_dir}')
    policy = policy_loader.load(policy_dir, ckpt_dir)

    params_module = import_params_module(args.params)
    with open(args.config, 'r') as f:
      config = json.load(f)
    ps: TrainParamsBase = params_module.TrainParams(config)
    env_args, env_kwargs = ps.eval_env_args_kwargs()
    env: PyEnvironment = ps.env_ctor(seed=args.seed, *env_args, **env_kwargs)

    # run
    total_return = 0.0
    for episode in range(args.num_episodes):
      print(f'Episode {episode+1} / {args.num_episodes}')
      time_step = env.reset()
      policy_state = policy.get_initial_state(env.batch_size or 1)
      frame_buffer = []
      episode_reward = time_step.reward.item()
      while True:
        if len(frame_buffer) % 100 == 0:
          print(f'step {len(frame_buffer)}')
        action_step = policy.action(time_step, policy_state)
        policy_state = action_step.state
        time_step = env.step(action_step.action)
        episode_reward += time_step.reward.item()
        frame_buffer.append(env.render())
        if time_step.is_last():
          break
      print(f'return = {episode_reward}, {len(frame_buffer)} steps')
      total_return += episode_reward
      filename = f'video{episode}.mp4'
      writer = imageio.get_writer(filename, fps=30, macro_block_size=None, ffmpeg_params = ['-s','960x720'])
      for f in frame_buffer:
        writer.append_data(f)
      writer.close()
      print(f'{filename} written')

    avg_return = total_return / args.num_episodes
    print(f'Average Return = {avg_return}')
    env.close()