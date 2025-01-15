import tensorflow as tf
from tensorflow.keras import layers
from utils import *


class Pix2Pix(tf.keras.Model):
    def __init__(self, generator, discriminator, lambda_L1=100.0):
        super(Pix2Pix, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_L1 = lambda_L1

        # We define loss objects here for convenience
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def compile(self, g_optimizer, d_optimizer):
        super(Pix2Pix, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def call(self, inputs, training=False):
        """
        Minimal call method for inference.
        It takes an input and returns the generator's output.
        """
        return self.generator(inputs, training=training)

    def generator_loss(self, disc_generated_output, gen_output, target):
        """
        1) Adversarial loss
        2) L1 loss
        """
        adv_loss = self.loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = adv_loss + (self.lambda_L1 * l1_loss)
        return total_gen_loss, adv_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_obj(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def train_step(self, batch_data):
        """
        batch_data is (input_images, target_images)
        where input_images = real_A, target_images = real_B
        """
        input_image, target_image = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # 1) Generate fake B
            fake_image = self.generator(input_image, training=True)

            # 2) Discriminate real (A,B) => real concat
            real_concat = tf.concat([input_image, target_image], axis=-1)
            disc_real = self.discriminator(real_concat, training=True)

            # 3) Discriminate fake (A, fakeB) => fake concat
            fake_concat = tf.concat([input_image, fake_image], axis=-1)
            disc_fake = self.discriminator(fake_concat, training=True)

            # 4) Compute losses
            d_loss = self.discriminator_loss(disc_real, disc_fake)
            g_loss, adv_loss, l1_loss = self.generator_loss(disc_fake, fake_image, target_image)

        # 5) Compute gradients and update
        generator_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        discriminator_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))

        return {
            "g_loss": g_loss,
            "d_loss": d_loss,
            "adv_loss": adv_loss,
            "l1_loss": l1_loss
        }

class Discriminator(tf.keras.Model):
    def __init__(self, df=64, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.df = df

        self.conv1 = layers.Conv2D(self.df, 4, strides=2, padding='same')
        self.conv2 = layers.Conv2D(self.df * 2, 4, strides=2, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(self.df * 4, 4, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(self.df * 8, 4, strides=1, padding='same')
        self.bn4 = layers.BatchNormalization()

        # Final layer: 1-dim output (real/fake)
        self.conv5 = layers.Conv2D(1, 4, strides=1, padding='same')

    def call(self, inputs, training=True):
        """
        inputs shape: (batch, 256, 256, 6) if A and B are concatenated along channels
        or (batch, 256, 256, 3) if only one side.
        For Pix2Pix, we often concat the input and target along channel axis => 6 channels.
        """
        x = lrelu(self.conv1(inputs))
        x = lrelu(self.bn2(self.conv2(x), training=training))
        x = lrelu(self.bn3(self.conv3(x), training=training))
        x = lrelu(self.bn4(self.conv4(x), training=training))
        x = self.conv5(x)  # No activation => raw logits
        return x


class Generator(tf.keras.Model):
    def __init__(self, gf=64, output_channels=3, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.gf = gf
        self.output_channels = output_channels

        # ENCODER
        self.down1 = self._block_down(self.gf, apply_batchnorm=False)  # 1
        self.down2 = self._block_down(self.gf * 2)  # 2
        self.down3 = self._block_down(self.gf * 4)  # 3
        self.down4 = self._block_down(self.gf * 8)  # 4
        self.down5 = self._block_down(self.gf * 8)  # 5
        self.down6 = self._block_down(self.gf * 8)  # 6
        self.down7 = self._block_down(self.gf * 8)  # 7
        self.down8 = self._block_down(self.gf * 8, apply_batchnorm=False)  # 8

        # DECODER
        self.up1 = self._block_up(self.gf * 8, apply_dropout=True)
        self.up2 = self._block_up(self.gf * 8, apply_dropout=True)
        self.up3 = self._block_up(self.gf * 8, apply_dropout=True)
        self.up4 = self._block_up(self.gf * 8)
        self.up5 = self._block_up(self.gf * 4)
        self.up6 = self._block_up(self.gf * 2)
        self.up7 = self._block_up(self.gf)

        self.last = layers.Conv2DTranspose(self.output_channels,
                                           kernel_size=4,
                                           strides=2,
                                           padding='same',
                                           activation='tanh')

    def _block_down(self, filters, apply_batchnorm=True):
        """One downsampling block: Conv -> (BatchNorm) -> LeakyReLU"""
        result = tf.keras.Sequential()
        result.add(layers.Conv2D(filters, 4, strides=2, padding='same', use_bias=False))
        if apply_batchnorm:
            result.add(layers.BatchNormalization())
        result.add(layers.LeakyReLU())
        return result

    def _block_up(self, filters, apply_dropout=False):
        """One upsampling block: ConvTranspose -> (BatchNorm) -> Dropout? -> ReLU"""
        result = tf.keras.Sequential()
        result.add(layers.Conv2DTranspose(filters, 4, strides=2, padding='same', use_bias=False))
        result.add(layers.BatchNormalization())
        if apply_dropout:
            result.add(layers.Dropout(0.5))
        result.add(layers.ReLU())
        return result

    def call(self, x, training=True):
        # Downsampling
        d1 = self.down1(x, training=training)  # (bs, 128, 128, gf)
        d2 = self.down2(d1, training=training)  # (bs, 64, 64, gf*2)
        d3 = self.down3(d2, training=training)  # (bs, 32, 32, gf*4)
        d4 = self.down4(d3, training=training)  # ...
        d5 = self.down5(d4, training=training)
        d6 = self.down6(d5, training=training)
        d7 = self.down7(d6, training=training)
        d8 = self.down8(d7, training=training)  # Bottleneck

        # Upsampling
        u1 = self.up1(d8, training=training)
        u1 = tf.concat([u1, d7], axis=-1)
        u2 = self.up2(u1, training=training)
        u2 = tf.concat([u2, d6], axis=-1)
        u3 = self.up3(u2, training=training)
        u3 = tf.concat([u3, d5], axis=-1)
        u4 = self.up4(u3, training=training)
        u4 = tf.concat([u4, d4], axis=-1)
        u5 = self.up5(u4, training=training)
        u5 = tf.concat([u5, d3], axis=-1)
        u6 = self.up6(u5, training=training)
        u6 = tf.concat([u6, d2], axis=-1)
        u7 = self.up7(u6, training=training)
        u7 = tf.concat([u7, d1], axis=-1)

        return self.last(u7)  # (bs, 256, 256, output_channels)

