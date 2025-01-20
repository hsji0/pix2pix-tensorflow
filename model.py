import tensorflow as tf
from tensorflow.keras import layers


def down_block(x, filters, kernel_size=4, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(filters, kernel_size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=not apply_batchnorm)(x)
    if apply_batchnorm:
        x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.2)(x)

def up_block(x, filters, kernel_size=4, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if apply_dropout:
        x = layers.Dropout(0.5)(x)
    return layers.ReLU()(x)

def build_generator(gf=64, output_channels=3, input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Downsampling
    d1 = down_block(inputs, gf, apply_batchnorm=False)    # (bs, 128, 128, gf)
    d2 = down_block(d1, gf*2)                           # (bs, 64, 64, gf*2)
    d3 = down_block(d2, gf*4)                           # (bs, 32, 32, gf*4)
    d4 = down_block(d3, gf*8)                           # (bs, 16, 16, gf*8)
    d5 = down_block(d4, gf*8)                           # (bs, 8, 8, gf*8)
    d6 = down_block(d5, gf*8)                           # (bs, 4, 4, gf*8)
    d7 = down_block(d6, gf*8)                           # (bs, 2, 2, gf*8)
    # d8 = down_block(d7, gf*8)                           # (bs, 1, 1, gf*8)

    # Upsampling with skip connections
    # u1 = up_block(d8, gf*8, apply_dropout=True)
    u1 = up_block(d7, gf*8, apply_dropout=True)
    u1 = layers.Concatenate()([u1, d6])
    u2 = up_block(u1, gf*8, apply_dropout=True)
    u2 = layers.Concatenate()([u2, d5])
    u3 = up_block(u2, gf*8, apply_dropout=True)
    u3 = layers.Concatenate()([u3, d4])
    u4 = up_block(u3, gf*8)
    u4 = layers.Concatenate()([u4, d3])
    u5 = up_block(u4, gf*4)
    u5 = layers.Concatenate()([u5, d2])
    u6 = up_block(u5, gf*2)
    u6 = layers.Concatenate()([u6, d1])
    # u7 = up_block(u6, gf)
    # u7 = layers.Concatenate()([u7, d1])

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                                  kernel_initializer=initializer, activation='tanh')
    outputs = last(u6)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Generator")

### for 320x240 input size
def build_generator_320(gf=64, output_channels=3, input_shape=(240, 320, 3)):
    inputs = layers.Input(shape=input_shape)

    # Downsampling
    d1 = down_block(inputs, gf, apply_batchnorm=False)
    d2 = down_block(d1, gf*2)
    d3 = down_block(d2, gf*4)
    d4 = down_block(d3, gf*8)
    d5 = down_block(d4, gf*8)
    d6 = down_block(d5, gf*8)

    # Upsampling with skip connections
    # u1 = up_block(d8, gf*8, apply_dropout=True)
    u1 = up_block(d6, gf*8, apply_dropout=True)
    u1 = layers.Concatenate()([u1, d5])
    u2 = up_block(u1, gf*8, apply_dropout=True)
    u2 = layers.Concatenate()([u2, d4])
    u3 = up_block(u2, gf*8, apply_dropout=True)
    u3 = layers.Concatenate()([u3, d3])
    u4 = up_block(u3, gf*8)
    u4 = layers.Concatenate()([u4, d2])
    u5 = up_block(u4, gf*4)
    u5 = layers.Concatenate()([u5, d1])

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                                  kernel_initializer=initializer, activation='tanh')
    outputs = last(u5)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Generator")


"""
This fully convolutional design allows the network to make decisions on local patches of the image, which is the essence of PatchGAN.
"""
def build_discriminator(df=64, input_shape=(256, 256, 6)):
    inputs = layers.Input(shape=input_shape)
    initializer = tf.random_normal_initializer(0., 0.02)

    x = layers.Conv2D(df, 4, strides=2, padding='same', kernel_initializer=initializer)(inputs)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(df*2, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(df*4, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(df*8, 4, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    outputs = layers.Conv2D(1, 4, strides=1, padding='same', kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Discriminator")
