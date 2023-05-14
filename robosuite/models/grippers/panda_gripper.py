"""
Gripper for Franka's Panda (has two fingers).
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class PandaGripperBase(GripperModel):
    """
    Gripper for Franka's Panda (has two fingers).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/panda_gripper.xml"), idn=idn)
        self._init_qpos = np.array([0.020833, -0.020833])

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return self._init_qpos

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_collision", "finger1_pad_collision"],
            "right_finger": ["finger2_collision", "finger2_pad_collision"],
            "left_fingerpad": ["finger1_pad_collision"],
            "right_fingerpad": ["finger2_pad_collision"],
        }


class PandaGripper(PandaGripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """

    multiplier_vector = np.array([-1.0, 1.0]) #preallocate

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

        inp = self.current_action + self.multiplier * self.speed * np.sign(action)
        #for i in range(len(inp)):
        #    inp[i] = min(max(inp[i], -1), 1)
        #self.current_action = inp
        inp = np.minimum(np.maximum(inp, -1.0, out=inp), 1.0, out=inp)
        self.current_action = inp

        #self.current_action = inp.clip(-1.0, 1.0, out=inp)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1

    @property
    def multiplier(self):
        return self.multiplier_vector
