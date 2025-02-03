import os
import shutil
import random

subfolders = ["tumor", "notumor"]

def split_images_into_folders(data_directory, train_ratio, test_ratio):
    # Create folders for training and testing
    train_dir = os.path.join(data_directory, 'train')
    test_dir = os.path.join(data_directory, 'test')

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for directory in [train_dir, test_dir]:
        for subfolder in subfolders:
            subfolder_dir = os.path.join(directory, subfolder)
            os.makedirs(subfolder_dir, exist_ok=True)

    # List all subdirectories in the main directory
    subdirectories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(data_directory, subdirectory)

        # List all image files in the subdirectory
        all_images = [f for f in os.listdir(subdirectory_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Shuffle the list of images randomly
        random.shuffle(all_images)

        # Calculate the number of images for each split
        total_images = len(all_images)
        train_split = int(train_ratio * total_images)

        # Split the images into training and testing sets
        train_images = all_images[:train_split]
        test_images = all_images[train_split:]

        # Move images to the respective folders
        for image in train_images:
            shutil.move(os.path.join(subdirectory_path, image), os.path.join(train_dir, subdirectory, image))

        for image in test_images:
            shutil.move(os.path.join(subdirectory_path, image), os.path.join(test_dir, subdirectory, image))

# Specify the path to the main directory containing subdirectories for each class
data_directory = 'datasetBinaryTumor/'

# Specify the split ratios
train_ratio = 0.8
test_ratio = 0.2

# Call the function to split images into training and testing sets
split_images_into_folders(data_directory, train_ratio, test_ratio)
