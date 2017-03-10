import numpy as np
import tensorflow as tf


def tf_random_normal_var(shape_vec, mean_val=1.0, stddev_val=0.02):
    return tf.Variable(tf.random_normal(shape=shape_vec,
                                        mean=mean_val,
                                        stddev=stddev_val))

def tf_truncated_normal_var(shape_vec, mean_val=1.0, stddev_val=0.02):
    return tf.Variable(tf.truncated_normal(shape=shape_vec,
                                           mean=mean_val,
                                           stddev=stddev_val))

def tf_constant_var(val, shape_vec):
    return tf.Variable(tf.constant(val, shape=shape_vec))
