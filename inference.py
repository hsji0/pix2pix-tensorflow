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

        # image = tf.io.decode_jpeg(image, channels=3)  # or decode_png if PNG
        # Convert to float [0, 1], then optionally scale to [-1, 1] if desired
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize
        image = tf.image.resize(image, IMAGE_SIZE)
        # You may want to scale to [-1,1], as often used in Pix2Pix
        image = (image * 2.0) - 1.0
        image = tf.expand_dims(image, 0)
        # image_path = os.path.join(SOURCE_DIRPATH, image_file)
        #
        # # Preprocess the input image and add batch dimension
        # input_image = preprocess_grayscale_image(image_path)
        # input_image = tf.expand_dims(input_image, axis=0)  # shape: (1, H, W, 3)

        # Run inference
        generated_output = generator(image, training=False)  # shape: (1, H, W, 3)

        # Take the first (and only) image from the batch
        output_image = generated_output[0]  # shape: (H, W, 3)

        # (Optional) Compute mean to check the average pixel intensity
        mean_val = tf.reduce_mean(output_image)
        print("Mean pixel value (in [-1,1]):", mean_val.numpy())

        #####################################################################
        # Post-process & save using TF ops, without converting to NumPy/PIL. #
        #####################################################################

        # 1) Scale from [-1, 1] → [0, 1]
        output_image = (output_image + 1.0) / 2.0
        output_image = tf.clip_by_value(output_image, 0.0, 1.0)

        # 2) Convert to uint8 ([0, 255])
        output_image_uint8 = tf.image.convert_image_dtype(output_image, dtype=tf.uint8)

        # 3) Encode as PNG
        encoded_png = tf.io.encode_png(output_image_uint8)

        # 4) Write the PNG bytes to a file
        if SAVE_IMAGES:
            base_name, _ = os.path.splitext(image_file)
            output_file_name = base_name + "_generated.png"
            output_path = os.path.join(OUTPUT_SAVE_DIRPATH, output_file_name)
            tf.io.write_file(output_path, encoded_png)
            print(f"Saved generated image to: {output_path}")

    print("Batch inference completed.")
