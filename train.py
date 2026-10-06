# Simple Training Script
# Run this to train your models quickly

import os
from sklearn.model_selection import train_test_split
from data_preprocessing_training import (
    load_and_preprocess_training_data, 
    prepare_data_for_cnn, 
    validate_data_structure, 
    create_data_directories
)
from cnn_other import train_cnn
from svm_hog_other import prepare_hog_features_for_training, train_svm

def main():
    print("🏥 Kidney Stone Detection - Training Script")
    print("=" * 50)
    
    # Check if data directories exist
    data_path = "data/train"
    
    if not os.path.exists(data_path):
        print("Creating data directory structure...")
        create_data_directories()
        print("\n⚠️  Please add your CT scan images to:")
        print("   • data/train/Normal/ (for normal scans)")
        print("   • data/train/Stone/ (for stone scans)")
        print("\nThen run this script again.")
        return
    
    # Validate data structure
    is_valid, message = validate_data_structure(data_path)
    if not is_valid:
        print(f"❌ Data validation failed: {message}")
        return
    
    print(f"✅ {message}")

    print("\n🚀 Starting data loading...")
    images, labels = load_and_preprocess_training_data(data_path)
    
    if len(images) == 0:
        print("❌ No images loaded. Exiting.")
        return

    # --- Train CNN ---
    print("\n[1/2] Training CNN Model...")
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = prepare_data_for_cnn(images, labels)
    
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    
    train_cnn(X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn, model_save_path="models/kidney_stone_cnn.h5")
    
    # --- Train SVM ---
    print("\n[2/2] Training SVM Model...")
    # For SVM, we need HOG features. We can reuse the split labels effectively if we are careful,
    # but to be perfectly consistent with original logic, we should split the HOG features similarly.
    # The original logic split `hog_features` separately with the same random state.
    
    hog_features = prepare_hog_features_for_training(images)
    
    # Use same random_state=42 to ensure same split as CNN (roughly, though data types differ)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
        hog_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_svm(X_train_svm, y_train_svm, X_test_svm, y_test_svm, model_save_path="models/kidney_stone_svm.pkl")
    
    print("\n🎉 Training completed!")
    print("You can now run the web interface: python app.py")

if __name__ == "__main__":
    main()

        print("\nThen run this script again.")
        return
    
    # Validate data structure
    is_valid, message = validate_data_structure(data_path)
    if not is_valid:
        print(f"❌ Data validation failed: {message}")
        return
    
    print(f"✅ {message}")
    
    # Initialize detector and train
    detector = KidneyStoneDetector()
    
    print("\n🚀 Starting training process...")
    detector.train_models(data_path)
    
    print("\n🎉 Training completed!")
    print("You can now run the web interface: python simple_app.py")

if __name__ == "__main__":
    main()
