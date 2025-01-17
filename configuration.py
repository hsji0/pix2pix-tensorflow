import os

# --- Common Directories ---
MODEL_SAVE_DIRPATH = r"D:\9.Pairset Color Transfer\MODEL\Flash Melt Mold\250117"

# --- Inference Configuration ---
GENERATOR_MODEL_PATH = os.path.join(MODEL_SAVE_DIRPATH, "generator_model")
DISCRIMINATOR_MODEL_PATH = os.path.join(MODEL_SAVE_DIRPATH, "discriminator_model")
SOURCE_DIRPATH = r"D:\9.Pairset Color Transfer\image\Flash Melt Mold\crop_0_test\GRAY"         # Folder containing input images
OUTPUT_SAVE_DIRPATH = r"D:\9.Pairset Color Transfer\image\Flash Melt Mold\crop_0_test\GEN"    # Folder to save results
SAVE_IMAGES = True                                   # Set to True to save generated images

# --- Training Configuration ---
NUM_EPOCHS = 1000
EARLYSTOP_PATIENCE = 15
# FOLDER_GRAY = r"D:\9.Pairset Color Transfer\IMAGE_GRAY"
# FOLDER_COLOR = r"D:\9.Pairset Color Transfer\IMAGE_COLOR"
FOLDER_GRAY = r"D:\9.Pairset Color Transfer\image\Flash Melt Mold\crop_0_gray"
FOLDER_COLOR = r"D:\9.Pairset Color Transfer\image\Flash Melt Mold\crop_0"
BATCH_SIZE = 4
IMAGE_SIZE = (512, 640)  #  480x640 대신 model downsample 때문..
CHECK_IMAGE_DIRPATH = r"D:\9.Pairset Color Transfer\check_epochs"
