"""
All zeros TF Policy
"""
import numpy as np
from tf_policy_actionserver_base import TFPolicyActionServerBase
from tf_agents.policies.scripted_py_policy import ScriptedPyPolicy
from tf_agents.specs.array_spec import sample_spec_nest
from tf_agents.trajectories import TimeStep

class ZeroPolicyActionServer(TFPolicyActionServerBase):
  def __init__(self, service_name='zero_policy', **kwargs):
    super(ZeroPolicyActionServer, self).__init__(service_name, **kwargs)
    action = np.zeros(self.action_spec.shape, self.action_spec.dtype)
    self.policy = ScriptedPyPolicy(self.time_step_spec, self.action_spec, [(np.inf, action)])
    self.policy_state = self.policy.get_initial_state()
    self.ts_template: TimeStep = sample_spec_nest(self.policy.time_step_spec, np.random)


if __name__ == '__main__':
  zp = ZeroPolicyActionServer()
  a = zp.execute_cb({'observation': {'header': 0}})