import os
import shutil
import random

def split_dataset(source_dir, train_dir, test_dir, train_ratio=0.7):
    # Ensure the train and test directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all class directories in the source directory
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        images = os.listdir(class_dir)

        # Shuffle the images
        random.shuffle(images)

        # Split images into train and test
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Create class directories in train and test folders
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        # Move images to the corresponding directories
        for img in train_images:
            src_path = os.path.join(class_dir, img)
            dst_path = os.path.join(train_dir, cls, img)
            shutil.move(src_path, dst_path)
        
        for img in test_images:
            src_path = os.path.join(class_dir, img)
            dst_path = os.path.join(test_dir, cls, img)
            shutil.move(src_path, dst_path)

        print(f'Processed class {cls}: {len(train_images)} images to train, {len(test_images)} images to test.')

# Usage example
source_directory = r'C:\VSCode\KAN_satellite\Dataset'
train_directory = r'C:\VSCode\KAN_satellite\train'
test_directory = r'C:\VSCode\KAN_satellite\test'

split_dataset(source_directory, train_directory, test_directory)
