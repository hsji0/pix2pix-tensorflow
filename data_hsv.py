import tensorflow as tf
import os

def load_image(img_path, target_size=(256, 256)):
    """Load, preprocess, and convert a single image to HSV."""
    # Read the image from file
    image_data = tf.io.read_file(img_path)
    if img_path.endswith('.bmp'):
       image = tf.io.decode_bmp(image_data, channels=3)
    elif img_path.endswith('.png'):
        image = tf.io.decode_png(image_data, channels=3)
    elif img_path.lower().endswith(('.jpg', '.jpeg')):
        image = tf.io.decode_jpeg(image_data, channels=3)
    else:
        raise ValueError("Unsupported image format")

    # Convert to float32 and scale to [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image to the target size
    image = tf.image.resize(image, target_size)

    # Convert from RGB to HSV color space
    image = tf.image.rgb_to_hsv(image)

    # Optionally: if you need to scale HSV to [-1, 1] (not typical for HSV)
    # image = (image * 2.0) - 1.0

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

