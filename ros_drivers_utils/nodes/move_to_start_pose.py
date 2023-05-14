#!/usr/bin/env python
"""
Move arm to pre-defined end-effector start poses
"""
import argparse
import json
import numpy as np
import os
import rospy
from softgrasp_ros.moveit_utils import ArmMover
from tf import transformations as tx

osp = os.path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poses_filename', default='start_poses.json')
    parser.add_argument('-n', type=int, default=0)
    args = parser.parse_args(rospy.myargv()[1:])
    
    rospy.init_node('move_to_start_pose', anonymous=True)
    
    filename = osp.expanduser(args.poses_filename)
    with open(filename, 'r') as f:
        poses = json.load(f)
    poses = np.asarray(poses['poses'])
    
    mover = ArmMover(display_trajectories=False, get_confirmation=False)
    mover.home()
    mover.eef_motion(poses[args.n], relative=True)
