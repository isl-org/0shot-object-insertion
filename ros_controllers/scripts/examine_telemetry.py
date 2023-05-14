#!/usr/bin/env python
import argparse
import rosbag
import numpy as np
import os.path as osp


if __name__ == '__main__':
 parser = argparse.ArgumentParser()
 parser.add_argument('--bag_filename', required=True)
 args = parser.parse_args()

 with rosbag.Bag(osp.expanduser(args.bag_filename)) as f:
   for ropic, msg, t in f.read_messages():
     obs = np.reshape(msg.observation, (8, 12))
     pass