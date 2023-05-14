"""
Gripper for the soft robotics gripper #TODO: allow for configurations
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class SoftRoboticsGripperBase(GripperModel):
    """
    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/mgtest.xml"), idn=idn)

    def format_action(self, action):
        #TODO: this is where we have to convert four actions into n signals
        return np.ones(60) * action * 1.4 #TODO: figure out exact mapping
        #TODO: 20 should come from xml directly.  use self._dof

    @property
    def init_qpos(self):
        return np.zeros(60) #TODO get this 20 from the xml directly.  use self._dof?

    #TODO: explicitly add collisions
    @property
    def _important_geoms(self):
        return {
            "left_finger_0": [f'finger{0}_{l}' for l in range(6)],
            "left_finger_1": [f'finger{2}_{l}' for l in range(6)],
            "right_finger_0": [f'finger{1}_{l}' for l in range(6)],
            "right_finger_1": [f'finger{3}_{l}' for l in range(6)],
        }


class SoftRoboticsGripper(SoftRoboticsGripperBase):
    """
    Modifies SoftRoboticsGripper to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action + np.ones(20) * self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 1 #TODO: do we really need this?

    @property
    def dof(self):
        return 1
