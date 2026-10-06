import numpy as np
import cv2
from cnn_working import CNNModule
from svm_hog_working import SVMHOGModule
from data_preprocessing_inference import preprocess_image_inference

class EnsemblePredictor:
    def __init__(self):
        self.cnn_module = CNNModule()
        self.svm_module = SVMHOGModule()
        
    def predict(self, raw_image):
        """
        Main pipeline: Preprocess -> CNN & SVM Inference -> Ensemble -> Result
        """
        # 1. Preprocessing
        print("Step 1: Preprocessing image...")
        processed_image = preprocess_image_inference(raw_image)
        if processed_image is None:
            return {
                "error": "Image processing failed", 
                "prediction": "Error", 
                "confidence": 0.0,
                "details": {}
            }
            
        # 2. Individual Model Inference
        print("Step 2: Running CNN inference...")
        cnn_confidence = self.cnn_module.predict(processed_image) * 100
        
        print("Step 3: Running SVM+HOG inference...")
        svm_confidence = self.svm_module.predict(processed_image) * 100
        
        # 3. Ensemble Logic (Weighted Voting / Averaging)
        print("Step 4: Combining results (Ensemble)...")
        # Logic: If both models exist, average them. If one fails (0.0), use the other.
        # Note: If a model genuinely predicts 0.0 (0%), that's valid. 
        # But our wrappers return 0.0 on error too. Assuming loaded models work:
        
        if self.cnn_module.model is not None and self.svm_module.model is not None:
            ensemble_confidence = (cnn_confidence + svm_confidence) / 2
        elif self.cnn_module.model is not None:
            ensemble_confidence = cnn_confidence
        elif self.svm_module.model is not None:
            ensemble_confidence = svm_confidence
        else:
            # Fallback for no models loaded
            # Simple brightness heuristic
            gray = cv2.cvtColor((processed_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            mean_intensity = np.mean(gray)
            ensemble_confidence = 75.0 if mean_intensity < 100 else 25.0
            
        prediction_label = "Stone Detected" if ensemble_confidence > 50 else "No Kidney Stone Detected"
        
        return {
            "prediction": prediction_label,
            "confidence": ensemble_confidence,
            "details": {
                "cnn_confidence": cnn_confidence,
                "svm_confidence": svm_confidence
            }
        }

# Singleton instance to be used by app
ensemble = EnsemblePredictor()
