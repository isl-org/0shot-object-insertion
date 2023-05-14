from geometry_msgs.msg import Pose
import tf.transformations as tx
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
import rospy
from transforms3d import quaternions as txq


def gpose2mat(pose):
  q = [
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z,
    pose.orientation.w,
  ]
  T = tx.quaternion_matrix(q)
  T[0, 3] = pose.position.x
  T[1, 3] = pose.position.y
  T[2, 3] = pose.position.z
  return T

def mat2gpose(T):
  pose = Pose()
  pose.position.x = T[0, 3]
  pose.position.y = T[1, 3]
  pose.position.z = T[2, 3]
  # q = tx.quaternion_from_matrix(T)
  q = txq.mat2quat(T[:3, :3])
  q = q[[1, 2, 3, 0]]
  pose.orientation.x = q[0]
  pose.orientation.y = q[1]
  pose.orientation.z = q[2]
  pose.orientation.w = q[3]
  return pose


class PoseBroadcaster(object):
  def __init__(self):
    self.broadcaster = tf2_ros.StaticTransformBroadcaster()

  def broadcast(self, gpose, parent, child):
    gpose = gpose.get_gpose()
    static_transformStamped = TransformStamped()
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = parent
    static_transformStamped.child_frame_id = child
    static_transformStamped.transform.translation.x = gpose.position.x
    static_transformStamped.transform.translation.y = gpose.position.y
    static_transformStamped.transform.translation.z = gpose.position.z
    static_transformStamped.transform.rotation.x = gpose.orientation.x
    static_transformStamped.transform.rotation.y = gpose.orientation.y
    static_transformStamped.transform.rotation.z = gpose.orientation.z
    static_transformStamped.transform.rotation.w = gpose.orientation.w
    self.broadcaster.sendTransform(static_transformStamped)


class GPose(object):
  def __init__(self, pose=None):
    if isinstance(pose, np.ndarray) and (pose.shape == (4, 4)):
      self._pose = np.copy(pose)
    elif isinstance(pose, Pose):
      self._pose = gpose2mat(pose)
    else:
      raise NotImplementedError

  def get_gpose(self):
    return mat2gpose(self._pose)
  
  def get_mat(self):
    return np.copy(self._pose)

  def set_from_mat(self, T):
    self._pose = T

  def set_from_gpose(self, gpose):
    self._pose = gpose2mat(gpose)

  # self * other
  def __mul__(self, other):
    return GPose(np.dot(self._pose, other.get_mat()))

  # other * self
  def __rmul__(self, other):
    return GPose(np.dot(other.get_mat(), self._pose))

  def inv(self):
    return GPose(np.linalg.inv(self._pose))


def get_K_T_EE():
  q = np.array([0.000, 0.000, -0.383, 0.924])
  q = q / np.linalg.norm(q)
  EE_T_K = tx.quaternion_matrix(q)
  EE_T_K[:3, 3] = [0.000, 0.000, 0.106]
  K_T_EE = GPose(np.linalg.inv(EE_T_K))
  return K_T_EE


def tfstamped_to_gpose(tfs):
  pose = Pose()
  pose.position.x = tfs.transform.translation.x
  pose.position.y = tfs.transform.translation.y
  pose.position.z = tfs.transform.translation.z
  pose.orientation.w = tfs.transform.rotation.w
  pose.orientation.x = tfs.transform.rotation.x
  pose.orientation.y = tfs.transform.rotation.y
  pose.orientation.z = tfs.transform.rotation.z
  return pose