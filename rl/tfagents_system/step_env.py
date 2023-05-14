import init_paths
from tfagents_system.train_params_base import TrainParamsBase
from tf_agents.policies.random_tf_policy import RandomTFPolicy
import json

def do_stepping(output):
  with open('config/simple.json', 'r') as f:
    config = json.load(f)
  tp = TrainParamsBase(config, dict(output=output, seed_sac_buffer=False, no_graphics=True))
  env = tp.env_ctor()
  policy = RandomTFPolicy(env.time_step_spec(), env.action_spec())

  N = 10000
  time_step = env.reset()
  policy_state = None
  for idx in range(N):
    if idx%100 == 0:
      print(f'Step {idx+1}/{N}')
    policy_step = policy.action(time_step, policy_state)
    time_step = env.step(policy_step.action)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--output')
  args = parser.parse_args()

  do_stepping(args.output)
