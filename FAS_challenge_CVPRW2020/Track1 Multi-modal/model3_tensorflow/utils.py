""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

from tensorflow.python.training.moving_averages import assign_moving_average
def batch_norm(x, training=True, eps=1e-05, decay=0.9, affine=True, name=None):
    train = training
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = x.get_shape().as_list()[-1:]
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        def mean_var_with_update():
            axis = [k for k in range(len(x.get_shape().as_list()) - 1)]
            mean, variance = tf.nn.moments(x, axis, name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))

        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x


def contrast_depth_conv(input, dilation_rate=1, op_name='contrast_depth'):
    ''' compute contrast depth in both of (out, label) '''
    assert (input.get_shape()[-1] == 1)

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]
    kernel_filter = np.array(kernel_filter_list, np.float32)
    kernel_filter = np.expand_dims(kernel_filter, axis=2)
    kernel_filter_tf = tf.constant(kernel_filter, dtype=tf.float32)

    if dilation_rate == 1:
        contrast_depth = tf.nn.conv2d(input, kernel_filter_tf, strides=[1, 1, 1, 1], padding='SAME', name=op_name)
    else:
        contrast_depth = tf.nn.atrous_conv2d(input, kernel_filter_tf,rate=dilation_rate, padding='SAME', name=op_name)

    return contrast_depth


def contrast_depth_loss(out, label):
    '''
    compute contrast depth in both of (out, label),
    then get the loss of them
    tf.atrous_convd match tf-versions: 1.4
    '''
    contrast_out = contrast_depth_conv(out, 1, 'contrast_out')
    contrast_label = contrast_depth_conv(label, 1, 'contrast_label')

    loss = tf.pow(contrast_out - contrast_label, 2)
    loss = tf.reduce_mean(loss)

    return loss


def L2_loss(out, label):
    loss = tf.pow(out - label, 2)
    loss = tf.reduce_mean(loss)

    return loss