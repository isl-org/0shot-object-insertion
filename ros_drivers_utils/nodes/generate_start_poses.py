#!/usr/bin/env python
import json
import random
import numpy as np
import tf.transformations as tx
import os.path as osp


def generate(N=10, filename=osp.join('src', 'softgrasp_ros', 'nodes', 'start_poses.json')):
  sag_offset = 0.03

  # change this transform if you move the target. Should not need to change anything else
  q = np.array([0.999, 0.024, -0.025, 0.004])
  q = q / np.linalg.norm(q)
  O_T_S_real = tx.quaternion_matrix(q)
  O_T_S_real[:3, 3] = [0.324, -0.214, 0.320]

  q = np.array([0.000, 0.000, -0.383, 0.924])
  q = q / np.linalg.norm(q)
  EE_T_K = tx.quaternion_matrix(q)
  EE_T_K[:3, 3] = [0.000, 0.000, 0.106]
  K_T_EE = np.linalg.inv(EE_T_K)
  
  O_T_K_real = np.array([
    0.9999903394858235, 0.0002200209726769293, -0.00013938079925271263, 0.0,
    0.0002200431582946771, -0.9999903365259444, 0.00015917583184345786, 0.0,
    -0.00013934711319233716, -0.0001592080291622649, -0.9999999776175926, 0.0,
    0.3069240504204701, -5.200133649371486e-05, 0.48419860126134584, 1.0
  ]).reshape((4, 4)).T
  O_T_EE_real = np.dot(O_T_K_real, K_T_EE)

  O_T_K_sim = np.array([
    [ 1.0,  0.0,  0.0,  4.35154012e-01],
    [ 0.0, -1.0,  0.0, -5.80971901e-06],
    [ 0.0,  0.0, -1.0,  5.24045370e-01],
    [ 0.0,  0.0,  0.0,  1.0]
  ])
  O_T_EE_sim = np.dot(O_T_K_sim, K_T_EE)

  O_T_S_sim_min = np.array([
    [1.0,  0.0,  0.0,  3.67016438e-01],
    [0.0, -1.0,  0.0, -1.99989882e-01],
    [0.0,  0.0, -1.0,  3.12253710e-01],
    [0.0, 0.0, 0.0,  1.0]
  ])
  EE_T_S_sim_min = np.dot(np.linalg.inv(O_T_EE_sim), O_T_S_sim_min)

  O_T_S_sim_max = np.array([
    [1.0,  0.0,  0.0, 4.67016438e-01],
    [0.0, -1.0,  0.0, 2.00010118e-01],
    [0.0,  0.0, -1.0, 3.12253710e-01],
    [0.0,  0.0,  0.0, 1.0]
  ])
  EE_T_S_sim_max = np.dot(np.linalg.inv(O_T_EE_sim), O_T_S_sim_max)

  O_T_EE_real_min = np.dot(O_T_S_real, np.linalg.inv(EE_T_S_sim_min))
  O_T_EE_real_min[2, 3] += sag_offset
  O_T_EE_real_max = np.dot(O_T_S_real, np.linalg.inv(EE_T_S_sim_max))
  O_T_EE_real_max[2, 3] += sag_offset

  # ArmMover applies O_T_2 = O_T_1 * delta => delta = inv(O_T_1) * O_T_2
  delta_O_T_EE_min = np.dot(np.linalg.inv(O_T_EE_real), O_T_EE_real_min)
  delta_O_T_EE_max = np.dot(np.linalg.inv(O_T_EE_real), O_T_EE_real_max)

  random.seed(7)
  xs = np.random.uniform(delta_O_T_EE_min[0, 3], delta_O_T_EE_max[0, 3], size=N)
  ys = np.random.uniform(delta_O_T_EE_min[1, 3], delta_O_T_EE_max[1, 3], size=N)
  zs = np.random.uniform(delta_O_T_EE_min[2, 3], delta_O_T_EE_max[2, 3], size=N)
  yaws = np.deg2rad(np.random.uniform(-20.0, 20.0, size=N))
  
  # # debug
  # xs = [delta_O_T_EE_min[0, 3], delta_O_T_EE_max[0, 3]]
  # ys = [delta_O_T_EE_min[1, 3], delta_O_T_EE_max[1, 3]]
  # zs = [delta_O_T_EE_min[2, 3], delta_O_T_EE_max[2, 3]]
  # yaws = [0, 0]
  
  poses = []
  for x, y, z, yaw in zip(xs, ys, zs, yaws):
    T = tx.euler_matrix(0.0, 0.0, yaw)
    T[:3, 3] = [x, y, z]
    poses.append([r.tolist() for r in T])
  
  with open(filename, 'w') as f:
    json.dump({'poses': poses}, f, indent=4)
  print('{:s} written'.format(filename))


if __name__ == '__main__':
  generate()