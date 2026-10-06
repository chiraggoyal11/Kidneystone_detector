import numpy as np
import cv2
import pickle
import os
try:
    from skimage.feature import hog
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. SVM features cannot be extracted.")

class SVMHOGModule:
    def __init__(self, model_path="kidney_stone_svm.pkl"):
        self.model = None
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """Load trained SVM model from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("✅ SVM model loaded successfully")
                return True
            else:
                print(f"⚠️ SVM model file not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading SVM model: {e}")
            return False

    def extract_features(self, image):
        """
        Extract HOG features via scikit-image.
        Args:
            image: numpy array (RGB) or grayscale
        """
        if not SKIMAGE_AVAILABLE:
            return np.zeros(324)
            
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                # Expecting normalized 0-1 float image from preprocessing, need uint8 0-255 for cv2
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image_uint8 = (image * 255).astype(np.uint8)
                else:
                    image_uint8 = image
                    
                gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
            else:
                if image.dtype == np.float32 or image.dtype == np.float64:
                    gray = (image * 255).astype(np.uint8)
                else:
                    gray = image

            # Resize to the same size used during training
            resized = cv2.resize(gray, (64, 64))
            
            # Calculate HOG features
            features = hog(resized, 
                          orientations=9, 
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), 
                          visualize=False,
                          feature_vector=True)
            return features
        except Exception as e:
            print(f"Error extracting HOG features: {e}")
            return np.zeros(324)

    def predict(self, processed_image):
        """
        Make a prediction using SVM+HOG.
        """
        if self.model is None:
            return 0.0
            
        try:
            features = self.extract_features(processed_image)
            features = features.reshape(1, -1)
            
            # Get probability for class 1 (Stone)
            prob = self.model.predict_proba(features)[0][1]
            return float(prob)
        except Exception as e:
            print(f"Error in SVM prediction: {e}")
            return 0.0
