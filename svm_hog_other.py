from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
import numpy as np
from svm_hog_working import SVMHOGModule

def prepare_hog_features_for_training(images):
    """
    Extract HOG features for a batch of images used for training.
    """
    svm_module = SVMHOGModule(model_path=None) # Helper instance
    hog_features = []
    print("Extracting HOG features for training data...")
    for img in images:
        feats = svm_module.extract_features(img)
        hog_features.append(feats)
    return np.array(hog_features)

def train_svm(X_train_hog, y_train, X_test_hog, y_test, model_save_path="kidney_stone_svm.pkl"):
    """
    Train the SVM model and save it.
    """
    print("Starting SVM training...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_hog, y_train)
    
    # Evaluate
    print("\nEvaluating SVM...")
    predictions = svm_model.predict(X_test_hog)
    print(classification_report(y_test, predictions))
    
    # Save
    print(f"Saving SVM model to {model_save_path}...")
    with open(model_save_path, 'wb') as f:
        pickle.dump(svm_model, f)
        
    return svm_model
