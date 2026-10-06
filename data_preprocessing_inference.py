import numpy as np
import cv2
from PIL import Image

def preprocess_image_inference(image, target_size=(150, 150)):
    """
    Preprocess a single image for model inference.
    
    Args:
        image: PIL Image or numpy array
        target_size: tuple, target dimensions (width, height)
        
    Returns:
        numpy array: Preprocessed image ready for model input (normalized, expanded dims if needed)
    """
    try:
        # Convert PIL to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Resize image
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    except Exception as e:
        print(f"Error in inference preprocessing: {e}")
        return None
