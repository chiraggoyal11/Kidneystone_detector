import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import os
from cnn_working import CNNModule

def train_cnn(X_train, y_train, X_test, y_test, model_save_path="kidney_stone_cnn.h5"):
    """
    Train the CNN model and save it.
    """
    cnn_module = CNNModule(model_path=model_save_path)
    model = cnn_module.build_model(input_shape=X_train.shape[1:])
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True
    )
    
    print("Starting CNN training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating CNN...")
    predictions = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, predictions))
    
    # Save
    print(f"Saving CNN model to {model_save_path}...")
    model.save(model_save_path)
    
    return history, model
