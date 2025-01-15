from model import *
from data import *
import os


NUM_EPOCHS = 1
MODEL_SAVE_DIRPATH = r"C:\Users\hsji\Downloads"
EARLYSTOP_PATIENCE = 15

if __name__ == "__main__":
    # 1. Create the dataset
    folderA = "D:\9.Pairset Color Transfer\IMAGE_GRAY"
    folderB = "D:\9.Pairset Color Transfer\IMAGE_COLOR"
    batch_size = 1
    image_size = (256, 256)

    train_ds = create_image_pairs_dataset(folderA, folderB,
                                          batch_size=batch_size,
                                          image_size=image_size)

    # 2. Create the generator & discriminator
    generator = Generator(gf=64, output_channels=3)
    discriminator = Discriminator(df=64)

    # 3. Create the Pix2Pix model
    pix2pix_model = Pix2Pix(generator, discriminator, lambda_L1=100.0)

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='g_loss',
        patience=EARLYSTOP_PATIENCE,
        mode='min',
        restore_best_weights=True
    )

    # 4. Compile with optimizers
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    pix2pix_model.compile(
        g_optimizer=generator_optimizer,
        d_optimizer=discriminator_optimizer,
    )

    # 5. Fit the model
    # (Note: For large datasets or multi-epoch, you may want model.fit(..., epochs=..., steps_per_epoch=...))
    pix2pix_model.fit(train_ds, epochs=NUM_EPOCHS,callbacks=[earlystop_callback])
    # I want to save entire model (in FINAL folder, model would be there)

    pix2pix_model.build(input_shape=[None, image_size[0], image_size[1], 3])  # None for batch size
    pix2pix_model.save(MODEL_SAVE_DIRPATH)


    # 6. (Optional) Save model weights
    model_save_dirpath = os.path.join(MODEL_SAVE_DIRPATH, "checkpoints")
    if not os.path.exists(model_save_dirpath):
        os.makedirs(model_save_dirpath)
    pix2pix_model.generator.save(os.path.join(model_save_dirpath, "generator.ckpt"))
    pix2pix_model.discriminator.save(os.path.join(model_save_dirpath, "discriminator.ckpt"))
