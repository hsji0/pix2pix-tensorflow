import tensorflow as tf
import os
import cv2
import numpy as np
from PIL import Image
import struct

Image.MAX_IMAGE_PIXELS = None

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha=alpha)

def convert_color(dirpath, save_dirpath):

    SUPPORTED_EXTENSIONS = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')

    for filename in os.listdir(dirpath):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            full_path = os.path.join(dirpath, filename)

            # Read image (as is). If grayscale, shape will be (H,W) or (H,W,1)
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Save in place
            full_path = os.path.join(save_dirpath, filename)
            cv2.imwrite(full_path, img)
            print(f"Converted {filename} saved")

def save_img(img_tensor, path):
    """
    Saves a single image (in [-1, 1] range) to 'path' as a PNG file using TF ops.
    """
    # Ensure we have a float32 Tensor
    if not isinstance(img_tensor, tf.Tensor):
        img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    # Scale [-1, 1] to [0, 1]
    img_tensor = (img_tensor + 1.0) / 2.0
    img_tensor = tf.clip_by_value(img_tensor, 0.0, 1.0)  # in case of any overshoot

    # Convert to uint8 (0-255)
    img_tensor_uint8 = tf.image.convert_image_dtype(img_tensor, dtype=tf.uint8)

    # Encode as PNG
    encoded_img = tf.io.encode_png(img_tensor_uint8)

    # Write the file
    tf.io.write_file(path, encoded_img)

# def save_intermediate_results(generator, sample_input, sample_target, epoch,
#                               output_dir=r"D:\9.Pairset Color Transfer\check_epochs"):
#     """
#     Generates and saves sample images (input, output, target) for inspection.
#     Args:
#         generator (tf.keras.Model): The generator model
#         sample_input (tf.Tensor): A batch of input images in [-1, 1] range
#         sample_target (tf.Tensor): A batch of target images in [-1, 1] range
#         epoch (int): Current epoch number
#         output_dir (str): Directory to save generated images
#     """
#
#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # Run the generator on the sample input
#     fake_image = generator(sample_input, training=False)
#
#     # Take just the first example in the batch
#     inp = sample_input[0]  # shape: (H, W, C)
#     tar = sample_target[0]
#     gen = fake_image[0]
#
#     # Now call our TF-based save_img
#     save_img(inp, os.path.join(output_dir, f"epoch_{epoch:03d}_input.png"))
#     save_img(tar, os.path.join(output_dir, f"epoch_{epoch:03d}_target.png"))
#     save_img(gen, os.path.join(output_dir, f"epoch_{epoch:03d}_generated.png"))
#
#     print(f"Saved intermediate results to '{output_dir}' for epoch {epoch}.")

def save_intermediate_results(generator, sample_input, sample_target, epoch, output_dir, num_images=30):
    """
    Generates and saves multiple sample images (input, output, target) for inspection.
    Args:
        generator (tf.keras.Model): The generator model
        sample_input (tf.Tensor): A batch of input images, shape (B, H, W, C), in [-1,1]
        sample_target (tf.Tensor): A batch of target images, shape (B, H, W, C), in [-1,1]
        epoch (int): Current epoch number
        output_dir (str): Directory to save generated images
        num_images (int): How many images from the batch to save
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run the generator on the entire batch
    fake_images = generator(sample_input, training=False)  # (B, H, W, C)

    # Decide how many images to save (bounded by the batch size)
    batch_size = sample_input.shape[0]
    num_to_save = min(num_images, batch_size)

    # Loop over the first `num_to_save` images in the batch
    for i in range(num_to_save):
        # Extract each image: (H, W, C)
        inp = sample_input[i]
        tar = sample_target[i]
        gen = fake_images[i]

        # Convert [-1,1] -> [0,1] if using tanh
        inp = (inp + 1) / 2.0
        tar = (tar + 1) / 2.0
        gen = (gen + 1) / 2.0

        # Clip to [0,1], convert to uint8
        inp = tf.clip_by_value(inp, 0.0, 1.0)
        tar = tf.clip_by_value(tar, 0.0, 1.0)
        gen = tf.clip_by_value(gen, 0.0, 1.0)

        inp_uint8 = tf.image.convert_image_dtype(inp, dtype=tf.uint8)
        tar_uint8 = tf.image.convert_image_dtype(tar, dtype=tf.uint8)
        gen_uint8 = tf.image.convert_image_dtype(gen, dtype=tf.uint8)

        # Encode as PNG
        encoded_inp = tf.io.encode_png(inp_uint8)
        encoded_tar = tf.io.encode_png(tar_uint8)
        encoded_gen = tf.io.encode_png(gen_uint8)

        # Construct file paths
        # We'll prefix them with epoch and index i
        input_path = os.path.join(output_dir, f"epoch_{epoch:03d}_img_{i:02d}_input.png")
        target_path = os.path.join(output_dir, f"epoch_{epoch:03d}_img_{i:02d}_target.png")
        gen_path = os.path.join(output_dir, f"epoch_{epoch:03d}_img_{i:02d}_generated.png")

        # Write files
        tf.io.write_file(input_path, encoded_inp)
        tf.io.write_file(target_path, encoded_tar)
        tf.io.write_file(gen_path, encoded_gen)

    print(f"Saved {num_to_save} images for epoch {epoch} in '{output_dir}'.")

def crop_and_save_large_image_with_offset(image_path, crop_height, crop_width, x_offset, y_offset,
                                          save_dir=r"D:\9.Pairset Color Transfer\image\Flash Melt Mold"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    with open(image_path, 'rb') as f:
        header_field, size, reserved1, reserved2, offset = struct.unpack('<2sI2HI', f.read(14))
        dib_header_size = struct.unpack('<I', f.read(4))[0]
        assert dib_header_size >= 40, "Unsupported DIB header size"

        # DIB header
        dib_header = f.read(dib_header_size - 4)
        width, height, _, bpp = struct.unpack('IiHH', dib_header[:12])

        # check_image_is_processable(width, height, channel=3)
        print("[INFO]")
        print(f"dib_header_size :{dib_header_size}")
        print(f"height :{height}  width:{width} bpp:{bpp}")
        assert bpp in (24, 32), "Only 24bpp and 32bpp supported"

        # Calculate row size and padding
        bytes_pp = bpp // 8  # bytes per pixel
        row_size_unpadded = width * bytes_pp  # row size without padding
        row_size = ((bpp * width + 31) // 32) * 4  # row size with padding
        padding = row_size - row_size_unpadded  # calculate padding

        # Initialize start position for cropping with offset
        x_start, y_start = x_offset, y_offset

        # Loop through the image, cropping and saving pieces
        img_num = 0
        while y_start < height:
            x_start = x_offset  # Reset x_start to x_offset after processing each row
            while x_start < width:
                # Calculate end position for cropping
                x_end = min(x_start + crop_width, width)
                y_end = min(y_start + crop_height, height)

                # Read each row and column in the crop area
                img_data = bytearray()
                for i in range(y_end - 1, y_start - 1, -1):
                    f.seek(offset + i * row_size + x_start * bytes_pp)
                    row_data = f.read((x_end - x_start) * bytes_pp)
                    img_data += row_data

                # Check sizes for debugging
                print(
                    f"Original size: {len(img_data)}, Expected: {(y_end - y_start) * (x_end - x_start) * bytes_pp}")

                # Reshape the data into an image
                img = np.frombuffer(img_data, dtype=np.uint8)
                if bpp == 32:
                    img = img.reshape((y_end - y_start, x_end - x_start, 4))
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                else:
                    img = img.reshape((y_end - y_start, x_end - x_start, 3))

                # Correct color representation to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                save_img_fullpath = os.path.join(save_dir, f'{crop_width}x{crop_height}_{img_num}.bmp')
                # save_img_fullpath = os.path.join(save_dir, f'{basename}_{crop_width}x{crop_height}_{img_num}.bmp')
                cv2.imwrite(save_img_fullpath, img)
                print(f"Saved as {save_img_fullpath}")

                x_start += crop_width
                img_num += 1
            y_start += crop_height

def crop_images(image_fullpath, crop_size, overlap, save_dirpath):
    """
    Args:
        image_fullpath: Path to the target image to be cropped.
        crop_size: Tuple (crop_width, crop_height) specifying the size of each crop.
        overlap: Number of pixels that adjacent crops will overlap.
        save_dirpath: Directory path where cropped images will be saved.
    Returns:
        None (Cropped images are saved to the specified directory.)
    """
    # Print debug information
    print(f"target image: {image_fullpath}")
    print(f"crop size: {crop_size}")
    print(f"overlap: {overlap}")
    print(f"save target path: {save_dirpath}")

    # Open the image
    image = Image.open(image_fullpath)
    width, height = image.size
    crop_width, crop_height = crop_size

    # Calculate step sizes based on desired overlap
    step_x = crop_width - overlap
    step_y = crop_height - overlap

    # Create the save directory if it does not exist
    os.makedirs(save_dirpath, exist_ok=True)

    # Determine x and y coordinates for the top-left corner of each crop
    # ensuring we cover the borders properly.
    if width >= crop_width:
        x_coords = list(range(0, width - crop_width + 1, step_x))
        if x_coords[-1] != width - crop_width:
            x_coords.append(width - crop_width)
    else:
        x_coords = [0]  # If image width smaller than crop width, use start 0.

    if height >= crop_height:
        y_coords = list(range(0, height - crop_height + 1, step_y))
        if y_coords[-1] != height - crop_height:
            y_coords.append(height - crop_height)
    else:
        y_coords = [0]  # If image height smaller than crop height, use start 0.

    # Counter for naming cropped images uniquely
    counter = 0

    # Iterate over all starting positions and crop the image
    for y in y_coords:
        for x in x_coords:
            # Define the bounding box for the crop
            box = (x, y, x + crop_width, y + crop_height)
            cropped_img = image.crop(box)

            # Construct a filename and save the cropped image
            save_path = os.path.join(save_dirpath, f"crop_{counter}.png")
            cropped_img.save(save_path)
            counter += 1

    print(f"Total crops saved: {counter}")


# if __name__ == "__main__":
#     image_fullpath = r"D:\9.Pairset Color Transfer\image\Flash Melt Mold\20000x20000_0.bmp"
#     crop_size = (640, 480)
#     overlap = 80
#     save_dirpath = r"D:\9.Pairset Color Transfer\image\Flash Melt Mold\crop_0"
#     crop_images(image_fullpath, crop_size, overlap, save_dirpath)

# if __name__ == "__main__":
#     img_fullpath = r"D:\4.Superpoint\data_\[3.2um][Flash 녹음, 곰팡이 불량].bmp"
#     crop_and_save_large_image_with_offset(img_fullpath, 20000, 20000, x_offset=20000, y_offset=20000)

if __name__ == "__main__":
    convert_color(dirpath=r"D:\9.Pairset Color Transfer\image\Flash Melt Mold\crop_0",
                  save_dirpath =r"D:\9.Pairset Color Transfer\image\Flash Melt Mold\crop_0_gray")  # color 는 BGR 기준
