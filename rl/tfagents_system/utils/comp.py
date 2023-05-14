import tensorflow as tf


def config_tf(n_interop_threads=1, n_intraop_threads=1):
  """
  configure TF threading
  """
  tf.config.threading.set_inter_op_parallelism_threads(n_interop_threads)
  tf.config.threading.set_intra_op_parallelism_threads(n_intraop_threads)
  
  # make GPUs invisible because we will train only on CPUs
  gpus = tf.config.list_physical_devices('GPU')
  if len(gpus) > 0:
    tf.config.set_visible_devices([], 'GPU')
  
  tf.compat.v1.enable_v2_behavior()