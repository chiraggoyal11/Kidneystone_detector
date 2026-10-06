import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

class CNNModule:
    def __init__(self, model_path="kidney_stone_cnn.h5"):
        self.model = None
        self.model_path = model_path
        self.load_model()
        
    def build_model(self, input_shape=(150, 150, 3)):
        """Build CNN model architecture"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model

    def load_model(self):
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print("✅ CNN model loaded successfully")
                return True
            else:
                print(f"⚠️ CNN model file not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading CNN model: {e}")
            return False

    def predict(self, processed_image):
        """
        Make a prediction on a single preprocessed image.
        Args:
            processed_image: numpy array of shape (height, width, channels)
        Returns:
            float: Probability of stone presence (0-1)
        """
        if self.model is None:
            print("CNN model is not loaded.")
            return 0.0
            
        try:
            # Add batch dimension if missing
            if len(processed_image.shape) == 3:
                img_batch = np.expand_dims(processed_image, axis=0)
            else:
                img_batch = processed_image
                
            prediction = self.model.predict(img_batch, verbose=0)
            return float(prediction[0][0])
        except Exception as e:
            print(f"Error in CNN prediction: {e}")
            return 0.0
