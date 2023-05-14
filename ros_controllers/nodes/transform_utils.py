import math
import numpy as np


def normalize_quat(q):
  return q / np.linalg.norm(q)


def axangle2quat(vec):
  """
  q: xyzw
  """
  angle = np.linalg.norm(vec)
  q = np.array([0.0, 0.0, 0.0, 1.0])
  if abs(angle) > 1e-3:
    axis = vec / angle
    q[:3] = axis * np.sin(angle / 2.)
    q[3] = np.cos(angle / 2.)
  return q


def quat2axangle(q):
  """
  q: xyzw
  """
  den = np.sqrt(1. - q[3] * q[3])
  if abs(den) < 1e-3:
    # This is (close to) a zero degree rotation, immediately return
    a = np.zeros(3)
  else:
    a = (q[:3] * 2. * math.acos(q[3])) / den
  return a


def quat_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions (q1 * q0).
    E.g.:
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True
    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion
        quaternion0 (np.array): (x,y,z,w) quaternion
    Returns:
        np.array: (x,y,z,w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float64,
    )


def clip_translation(dpos, limit):
  input_norm = np.linalg.norm(dpos)
  return dpos*limit/input_norm if input_norm>limit else dpos


def clip_axangle_rotation(rot, limit):
  a = np.linalg.norm(rot)
  return rot*limit/a if a>limit else rot