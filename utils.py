import tensorflow as tf
from tensorflow.keras import layers
import os
import cv2
import numpy as np


def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha=alpha)

class ReflectionPadding2D(layers.Layer):
    """
    Optional: Some people prefer reflection padding for better style transfers.
    For Pix2Pix, 'same' padding with Conv2D is typically enough.
    This is just an example if you want reflection padding.
    """
    def __init__(self, padding=(1,1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        pad_w, pad_h = self.padding
        return tf.pad(
            input_tensor,
            [[0,0], [pad_h, pad_h], [pad_w, pad_w], [0,0]],
            'REFLECT'
        )

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding})
        return config


def convert_color(dirpath=r"D:\9.Pairset Color Transfer\IMAGEB",
                  color=(0, 255, 0)):
    """
    Convert any grayscale image in 'dirpath' into a tinted color version,
    using the specified 'color' (BGR) to tint the image.

    Args:
        dirpath (str): Path to the folder containing images.
        color (tuple): BGR color for tinting, e.g. (B, G, R).
                       Default is (0, 255, 0) => green.

    Supported extensions: .bmp, .jpg, .jpeg, .png, .tif, .tiff
    """
    SUPPORTED_EXTENSIONS = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')

    for filename in os.listdir(dirpath):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            full_path = os.path.join(dirpath, filename)

            # Read image (as is). If grayscale, shape will be (H,W) or (H,W,1)
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"Could not read '{filename}'. Skipping...")
                continue

            # Check if it's already color (3 channels or 4 channels).
            # If it's grayscale, it often has 2 dims or the 3rd dim == 1.
            if len(img.shape) == 2:
                # shape: (H, W), definitely grayscale
                h, w = img.shape
                # Convert to float in [0,1]
                gray_float = img.astype(np.float32) / 255.0
                # Expand to 3 channels
                gray_float_3d = np.stack([gray_float, gray_float, gray_float], axis=-1)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                # shape: (H, W, 1)
                h, w, c = img.shape
                gray_float = img[:, :, 0].astype(np.float32) / 255.0
                gray_float_3d = np.stack([gray_float, gray_float, gray_float], axis=-1)
            else:
                # Already color (e.g., (H,W,3) or (H,W,4))
                print(f"'{filename}' is already color or has alpha channel. Skipping tint.")
                continue

            # color is a tuple of BGR in [0..255]
            color_arr = np.array(color, dtype=np.float32)

            # Multiply each pixel by the chosen color
            #   gray_float_3d: shape (H, W, 3) in [0..1]
            #   color_arr: shape (3,) in [0..255]
            colored = gray_float_3d * color_arr

            # Clip and convert back to uint8 in [0..255]
            colored = np.clip(colored, 0, 255).astype(np.uint8)

            # Save in place
            cv2.imwrite(full_path, colored)
            print(f"Converted grayscale '{filename}' to color with tint {color}.")

if __name__ == "__main__":
    convert_color()
