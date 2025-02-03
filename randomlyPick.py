import os
import shutil
import random

def move_random_images(source_dir, target_dir, num_images):
    """
    Moves a specified number of random images from the source directory
    to the target directory.

    Parameters:
        source_dir (str): The path to the source directory.
        target_dir (str): The path to the target directory.
        num_images (int): The number of images to move.
    """

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get a list of all image files in the source directory
    # This example assumes files are JPEGs; adjust the extension if necessary.
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Check if the folder has enough images
    if len(files) < num_images:
        print(f"Not enough images in {source_dir}. Only {len(files)} available.")
        return

    # Select random images
    selected_files = random.sample(files, num_images)

    # Move selected images to target directory
    for file in selected_files:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(target_dir, file)
        shutil.move(src_path, dst_path)
        print(f"Moved {file} to {target_dir}")

# Example usage
source_directory = './old_dataset/mri/Healthy'
target_directory = './dataset/mri/Healthy'
number_of_images = 2000  # Change to how many images you want to move

move_random_images(source_directory, target_directory, number_of_images)
