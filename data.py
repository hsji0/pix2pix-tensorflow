import tensorflow as tf


def load_image(img_path, target_size=(256, 256)):
    """Load and preprocess a single image."""
    # Read the image
    image = tf.io.read_file(img_path)
    if img_path.endswith('.bmp'):
       image = tf.io.decode_bmp(image, channels=3)
    elif img_path.endswith('.png'):
        image = tf.io.decode_png(image, channels=3)
    elif img_path.lower().endswith(('.jpg', '.jpeg')):
        image = tf.io.decode_jpeg(image, channels=3)
    else:
        raise ValueError("Unsupported image format")

    # image = tf.io.decode_jpeg(image, channels=3)  # or decode_png if PNG
    # Convert to float [0, 1], then optionally scale to [-1, 1] if desired
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize
    image = tf.image.resize(image, target_size)
    # You may want to scale to [-1,1], as often used in Pix2Pix
    image = (image * 2.0) - 1.0
    return image

def create_image_pairs_dataset(folderA, folderB, batch_size=1, image_size=(256, 256)):
    # Collect filenames in each folder. Sort them so that the order matches.
    # We assume the same set of filenames exist in folderA and folderB.
    a_filenames = sorted([f for f in os.listdir(folderA) if f.lower().endswith(('jpg','png','jpeg','bmp'))])
    b_filenames = sorted([f for f in os.listdir(folderB) if f.lower().endswith(('jpg','png','jpeg','bmp'))])

    # Create the absolute paths
    a_paths = [os.path.join(folderA, fname) for fname in a_filenames]
    b_paths = [os.path.join(folderB, fname) for fname in b_filenames]

    # Zip them together
    paired_paths = list(zip(a_paths, b_paths))

    def gen():
        for (pathA, pathB) in paired_paths:
            imgA = load_image(pathA, target_size=image_size)
            imgB = load_image(pathB, target_size=image_size)
            yield (imgA, imgB)

    # Create a tf.data.Dataset from this generator
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([image_size[0], image_size[1], 3], [image_size[0], image_size[1], 3])
    )

    # Batch and shuffle, if desired
    dataset = dataset.shuffle(buffer_size=len(paired_paths)).batch(batch_size)
    return dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from configuration import *

    # Assuming you've already created your dataset
    dataset = create_image_pairs_dataset(FOLDER_GRAY, FOLDER_COLOR,
                                      batch_size=BATCH_SIZE,
                                      image_size=IMAGE_SIZE)

    save_dir = r"C:\Users\hsji\Downloads\temp"
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over the dataset and save image pairs
    for idx, (imgA_batch, imgB_batch) in enumerate(dataset.take(1)):  # Use .take(1) or another limit if needed
        # Process each image in the batch
        for i in range(imgA_batch.shape[0]):
            imgA = imgA_batch[i]
            imgB = imgB_batch[i]

            # Convert from [-1, 1] back to [0, 1] range
            dispA = (imgA + 1) / 2.0
            dispB = (imgB + 1) / 2.0

            # Convert to [0, 255] uint8 for saving
            dispA_uint8 = tf.image.convert_image_dtype(dispA, dtype=tf.uint8)
            dispB_uint8 = tf.image.convert_image_dtype(dispB, dtype=tf.uint8)

            # Encode as PNG or JPEG
            encodedA = tf.io.encode_png(dispA_uint8)
            encodedB = tf.io.encode_png(dispB_uint8)

            # Create file paths
            file_path_A = os.path.join(save_dir, f"batch{idx}_img{i}_A.png")
            file_path_B = os.path.join(save_dir, f"batch{idx}_img{i}_B.png")

            # Write the files
            tf.io.write_file(file_path_A, encodedA)
            tf.io.write_file(file_path_B, encodedB)

