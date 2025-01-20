import configuration
from model import build_generator, build_discriminator
from data import create_image_pairs_dataset
from configuration import *
from utils import *


# 1. Create the dataset
train_ds = create_image_pairs_dataset(FOLDER_GRAY, FOLDER_COLOR,
                                      batch_size=BATCH_SIZE,
                                      image_size=IMAGE_SIZE)

# 2. Build functional models for Generator and Discriminator
generator = build_generator(gf=64, output_channels=3, input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
# Discriminator expects concatenated images along channels (6 channels total)
discriminator = build_discriminator(df=64, input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],6))

# 3. Set up optimizers and loss
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target, lambda_L1=100.0):
    adv_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = adv_loss + (lambda_L1 * l1_loss)
    return total_gen_loss, adv_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

@tf.function
def train_step(input_image, target_image):
    with tf.GradientTape(persistent=True) as tape:
        # Generate fake image from input
        fake_image = generator(input_image, training=True)

        # Concatenate input with real and fake outputs for discriminator
        real_concat = tf.concat([input_image, target_image], axis=-1)
        fake_concat = tf.concat([input_image, fake_image], axis=-1)

        # Discriminator forward passes
        disc_real = discriminator(real_concat, training=True)
        disc_fake = discriminator(fake_concat, training=True)

        # Compute losses
        d_loss = discriminator_loss(disc_real, disc_fake)
        g_loss, adv_loss, l1_loss = generator_loss(disc_fake, fake_image, target_image)

    # Compute gradients
    generator_grads = tape.gradient(g_loss, generator.trainable_variables)
    discriminator_grads = tape.gradient(d_loss, discriminator.trainable_variables)

    # Apply gradients
    generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))

    return {"g_loss": g_loss, "d_loss": d_loss, "adv_loss": adv_loss, "l1_loss": l1_loss}



# Create checkpoint and manager
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator,
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer)

manager = tf.train.CheckpointManager(checkpoint,
                                     directory='checkpoints/best_model',
                                     max_to_keep=1)

# 4. Training loop with Early Stopping
best_loss = float('inf')
patience_counter = 0
# Fetch one sample batch for saving
sample_input, sample_target = next(iter(train_ds))

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    epoch_g_loss_sum = 0.0
    steps_count = 0

    for step, (input_image, target_image) in enumerate(train_ds):
        losses = train_step(input_image, target_image)
        epoch_g_loss_sum += losses["g_loss"].numpy()
        steps_count += 1

        if step % 100 == 0:
            print(f"Step {step}, Generator Loss: {losses['g_loss'].numpy()}, Discriminator Loss: {losses['d_loss'].numpy()}")


    save_intermediate_results(
        generator,
        sample_input,
        sample_target,
        epoch + 1,
        output_dir=configuration.CHECK_IMAGE_DIRPATH,
        num_images=configuration.BATCH_SIZE
    )

    # Compute average generator loss for the epoch
    avg_epoch_g_loss = epoch_g_loss_sum / steps_count
    print(f"Average Generator Loss for epoch {epoch+1}: {avg_epoch_g_loss}")

    # Check improvement for early stopping
    if avg_epoch_g_loss < best_loss:
        best_loss = avg_epoch_g_loss
        patience_counter = 0
        # Save a checkpoint
        save_path = manager.save()
        print(f"New best model saved at: {save_path}")
    else:
        patience_counter += 1

    if patience_counter >= EARLYSTOP_PATIENCE:
        print("Early stopping triggered. Restoring best checkpoint...")
        # Restore from the best checkpoint
        checkpoint.restore(manager.latest_checkpoint)
        break


if not os.path.exists(GENERATOR_MODEL_PATH):
    os.makedirs(GENERATOR_MODEL_PATH)
if not os.path.exists(DISCRIMINATOR_MODEL_PATH):
    os.makedirs(DISCRIMINATOR_MODEL_PATH)

# 5. Save the models
generator.save(GENERATOR_MODEL_PATH)
discriminator.save(DISCRIMINATOR_MODEL_PATH)
