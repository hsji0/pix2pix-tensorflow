import os
import tensorflow as tf
import numpy as np
from configuration import *

def preprocess_grayscale_image(image_path, target_size=IMAGE_SIZE):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)  # decode to 3 channels if needed
    image = tf.image.resize(image, target_size)
    image = tf.image.convert_image_dtype(image, tf.float32)  # scale to [0,1]
    image = (image * 2.0) - 1.0  # scale to [-1, 1]
    return image

if __name__ == "__main__":
    # Create output directory if saving is enabled
    if SAVE_IMAGES:
        os.makedirs(OUTPUT_SAVE_DIRPATH, exist_ok=True)

    # Load the pretrained generator model once
    generator = tf.keras.models.load_model(GENERATOR_MODEL_PATH)

    # Gather list of image files from the source directory
    valid_extensions = (".bmp", ".jpg", ".jpeg", ".png")
    image_files = [
        f for f in os.listdir(SOURCE_DIRPATH)
        if f.lower().endswith(valid_extensions)
    ]
    image_files = [os.path.join(SOURCE_DIRPATH, file_name) for file_name in image_files]

    for image_file in image_files:
        # print(image_file)
        image = tf.io.read_file(image_file)

        if image_file.endswith('bmp'):
            image = tf.io.decode_bmp(image, channels=3)
        elif image_file.endswith('png'):
            image = tf.io.decode_png(image, channels=3)
        elif image_file.endswith('jpeg'):
            image = tf.io.decode_jpeg(image, channels=3)

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.image.rgb_to_hsv(image)
        image = tf.expand_dims(image, 0)


        generated_output = generator(image, training=False)  # shape: (1, H, W, 3)

        # Take the first (and only) image from the batch
        output_image = generated_output[0]  # shape: (H, W, 3)

        # (Optional) Compute mean to check the average pixel intensity
        mean_val = tf.reduce_mean(output_image)
        output_image = tf.clip_by_value(output_image, 0.0, 1.0)

        image = tf.image.hsv_to_rgb(output_image)
        # 2) Convert to uint8 ([0, 255])
        image = tf.image.resize(image, TARGET_SIZE)
        output_image_uint8 = tf.image.convert_image_dtype(image, dtype=tf.uint8)

        # 3) Encode as PNG
        encoded_png = tf.io.encode_png(output_image_uint8)

        # 4) Write the PNG bytes to a file
        if SAVE_IMAGES:
            base_name, _ = os.path.splitext(os.path.basename(image_file))
            output_file_name = base_name + "_generated.png"
            output_path = os.path.join(OUTPUT_SAVE_DIRPATH, output_file_name)
            tf.io.write_file(output_path, encoded_png)
            print(f"Saved generated image to: {output_path}")

    print("Batch inference completed.")
