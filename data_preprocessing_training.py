import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_and_preprocess_training_data(data_dir, image_size=(150, 150)):
    """
    Load and preprocess data for training.
    
    Args:
        data_dir: str, path to data directory containing 'Normal' and 'Stone' subfolders
        image_size: tuple, target image size
        
    Returns:
        tuple: (images, labels) arrays
    """
    images = []
    labels = []
    
    # Load Normal images
    normal_dir = os.path.join(data_dir, 'Normal')
    if os.path.exists(normal_dir):
        print(f"Loading Normal images from {normal_dir}...")
        for filename in os.listdir(normal_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(normal_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, image_size)
                    images.append(img_resized)
                    labels.append(0)  # 0 for Normal
    
    # Load Stone images
    stone_dir = os.path.join(data_dir, 'Stone')
    if os.path.exists(stone_dir):
        print(f"Loading Stone images from {stone_dir}...")
        for filename in os.listdir(stone_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(stone_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, image_size)
                    images.append(img_resized)
                    labels.append(1)  # 1 for Stone
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def prepare_data_for_cnn(images, labels, test_size=0.2, random_state=42):
    """
    Prepare data specifically for CNN training (normalization and splitting).
    """
    # Normalize images
    images_normalized = images.astype('float32') / 255.0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images_normalized, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    return X_train, X_test, y_train, y_test

def validate_data_structure(data_path):
    """Validate that data directory has correct structure"""
    required_dirs = ['Normal', 'Stone']
    
    if not os.path.exists(data_path):
        return False, f"Data directory not found: {data_path}"
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = os.path.join(data_path, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        return False, f"Missing directories: {missing_dirs}"
    
    # Check if directories have images
    total_images = 0
    for dir_name in required_dirs:
        dir_path = os.path.join(data_path, dir_name)
        image_count = len([f for f in os.listdir(dir_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        total_images += image_count
        
        if image_count == 0:
            return False, f"No images found in {dir_name} directory"
    
    return True, f"Data structure valid. Found {total_images} total images."

def create_data_directories():
    """Create required data directory structure"""
    dirs_to_create = [
        'data/train/Normal',
        'data/train/Stone',
        'data/test/Normal', 
        'data/test/Stone',
        'models'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Created data directory structure:")
    print("data/")
    print("├── train/")
    print("│   ├── Normal/")
    print("│   └── Stone/")
    print("└── test/")
    print("    ├── Normal/")
    print("    └── Stone/")
    print("\nPlace your CT scan images in the appropriate folders.")
