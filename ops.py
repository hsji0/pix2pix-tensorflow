import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    """
    A simple block that does Conv -> optional BN -> optional LeakyReLU/ReLU
    """
    def __init__(self,
                 filters,
                 kernel_size=4,
                 strides=2,
                 padding="same",
                 apply_batchnorm=True,
                 apply_relu=True,
                 leaky_relu=False,
                 stddev=0.02,
                 name=None):
        super(ConvBlock, self).__init__(name=name)
        self.apply_batchnorm = apply_batchnorm
        self.apply_relu = apply_relu
        self.leaky_relu = leaky_relu

        # Convolution
        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=tf.random_normal_initializer(stddev=stddev),
            use_bias=not apply_batchnorm  # If using BN, bias can be omitted
        )

        # Optional BatchNorm
        if self.apply_batchnorm:
            self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=True):
        x = self.conv(x)
        if self.apply_batchnorm:
            x = self.bn(x, training=training)

        if self.apply_relu:
            if self.leaky_relu:
                x = tf.nn.leaky_relu(x, alpha=0.2)
            else:
                x = tf.nn.relu(x)
        return x


class DeconvBlock(tf.keras.layers.Layer):
    """
    Deconv (Conv2DTranspose) -> optional BN -> optional Dropout -> ReLU
    """
    def __init__(self,
                 filters,
                 kernel_size=4,
                 strides=2,
                 padding="same",
                 apply_batchnorm=True,
                 apply_dropout=False,
                 dropout_rate=0.5,
                 stddev=0.02,
                 name=None):
        super(DeconvBlock, self).__init__(name=name)
        self.apply_batchnorm = apply_batchnorm
        self.apply_dropout = apply_dropout

        self.deconv = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=tf.random_normal_initializer(stddev=stddev),
            use_bias=not apply_batchnorm
        )

        if self.apply_batchnorm:
            self.bn = tf.keras.layers.BatchNormalization()

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=True):
        x = self.deconv(x)
        if self.apply_batchnorm:
            x = self.bn(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)  # Typically ReLU in Pix2Pix decoders
        return x


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


