from utils import *


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


