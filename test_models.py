"""
Test script to verify the trained models work correctly
"""

import os
import numpy as np
from ensemble_learning import ensemble

def test_models():
    print("🧪 Testing Kidney Stone Detection Models")
    print("=" * 50)
    
    # Check if models loaded
    cnn_loaded = ensemble.cnn_module.model is not None
    svm_loaded = ensemble.svm_module.model is not None
    
    print(f"CNN Model loaded: {'✅' if cnn_loaded else '❌'}")
    print(f"SVM Model loaded: {'✅' if svm_loaded else '❌'}")
    
    if not (cnn_loaded or svm_loaded):
        print("\n❌ No models loaded! Please run training first:")
        print("   python train.py")
        return
    
    # Test with a sample image from test set
    import cv2
    
    test_image_paths = []
    
    # Look for test images
    normal_test_dir = "data/test/Normal"
    stone_test_dir = "data/test/Stone"
    
    if os.path.exists(normal_test_dir):
        normal_files = [f for f in os.listdir(normal_test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if normal_files:
            test_image_paths.append((os.path.join(normal_test_dir, normal_files[0]), "Normal"))
    
    if os.path.exists(stone_test_dir):
        stone_files = [f for f in os.listdir(stone_test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if stone_files:
            test_image_paths.append((os.path.join(stone_test_dir, stone_files[0]), "Stone"))
    
    # Test predictions
    if test_image_paths:
        print(f"\n🔬 Testing predictions on {len(test_image_paths)} sample images...")
        
        for image_path, expected in test_image_paths:
            print(f"\nTesting: {os.path.basename(image_path)} (Expected: {expected})")
            
            # Read image using cv2 to simulate input
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            result = ensemble.predict(img)
            
            if "error" not in result:
                prediction = result['prediction']
                confidence = result['confidence']
                details = result['details']
                cnn_conf = details.get('cnn_confidence', 0)
                svm_conf = details.get('svm_confidence', 0)
                
                print(f"  Prediction: {prediction}")
                print(f"  Confidence: {confidence:.2f}%")
                print(f"  CNN: {cnn_conf:.2f}%, SVM: {svm_conf:.2f}%")
                
                # Check if prediction matches expected
                # Note: original code used "Normal", new code uses "No Kidney Stone Detected"
                expected_pred = "Stone Detected" if expected == "Stone" else "No Kidney Stone Detected"
                status = "✅ CORRECT" if prediction == expected_pred else "❌ INCORRECT"
                print(f"  Status: {status}")
            else:
                print(f"  ❌ Prediction failed: {result.get('error')}")
    else:
        print("\n⚠️  No test images found in data/test/ directories")
        
    print("\n🎉 Model testing completed!")

if __name__ == "__main__":
    test_models()
