import numpy as np
import tensorflow as tf
import tf_helper as th

def batch_norm(x, beta, gamma, if_train, scope="bn", decay=0.9, eps=1e-5):
    with tf.variable_scope(scope):
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name="moments")
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identify(batch_mean), tf.identify(batch_var)
        mean, var = tf.cond(if_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed

def generative_model(z_dim,
                     image_size,
                     image_channel):
    params = {
        "W_1": th.tf_truncated_normal_var([z_dim, 2*image_size*image]),
        "b_1": th.tf_constant_var(0.0, [2*image_size*image_size]),
        "beta_1": th.tf_constant_var(0.0, [512]),
        "gamma_1": th.tf_random_normal_var([512]),
        
        "W_2": th.tf_truncated_normal_var([5, 5, 256, 512]),
        "b_2": th.tf_constant_var(0.0, [256]),
        "beta_2": th.tf_constant_var(0.0, [256]),
        "gemma_2": th.tf_random_normal_var([256]),

        "W_3": th.tf_truncated_normal_var([5, 5, 128, 256]),
        "b_3": th.tf_constant_var(0.0, [128]),
        "beta_3": th.tf_constant_var(0.0, [128]),
        "gamma_3": th.tf_random_normal_var([128]),

        "W_4": th.tf_truncated_normal_var([5, 5, 64, 128]),
        "b_4": th.tf_constant_var(0.0, [64]),
        "beta_4": th.tf_constant_var(0.0, [64]),
        "gamma_4": th.tf_random_normal_var([64]),

        "W_5": th.tf_truncated_normal_var([5, 5, image_channel, 64]),
        "b_5": th.tf_constant_var(0.0, shape=[image_channel])
    }
    return params

def generative_flow(noise, params, image_size):
    train_phrase = tf.placeholder(tf.bool)
    f1 = tf.matmul(noise, params["W_1"]) + params["b_1"]
    f1 = tf.reshape(f1, [-1, image_size//16, image_size//16, 512])
    f1 = batch_norm(f1, params["beta_1"], params["gamma_1"], train_phrase, scope="bn_1")
    f1 = tf.nn.relu(f1, name="relu_1")

    f2 = tf.conv2d_transpose(f1,
                             params["W_2"],
                             output_shape=tf.pack([tf.shape(f1)[0], image_size//8, image_size//8, 256]),
                             strides=[1,2,2,1],
                             padding="SAME")
    f2 = tf.nn.bias_add(f2, params["b_2"])
    f2 = batch_norm(f2, params["beta_2"], params["gamma_2"], train_phrase, scope="bn_2")
    f2 = tf.nn.relu(f2, name="relu_2")

    f3 = tf.conv2d_transpose(f2,
                             params["W_3"],
                             output_shape=tf.pack([tf.shape(f2)[0], image_size//4, image_size//4, 128]),
                             strides=[1,2,2,1],
                             padding="SAME")
    f3 = tf.nn.bias_add(f3, params["b_3"])
    f3 = batch_norm(f3, params["beta_3"], params["gamma_3"], train_phrase, scope="bn_3")
    f3 = tf.nn.relu(f3, name="relu_3")

    return
