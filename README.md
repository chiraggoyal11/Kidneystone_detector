# Kidney Stone Detection System

An advanced AI-powered diagnostic tool developed for the automated detection of kidney stones in CT scans using Ensemble Machine Learning techniques.

## 🌟 Project Overview

This project implements a robust medical image analysis system that combines:
1. **Deep Learning (CNN)**: Feature extraction and pattern recognition.
2. **Machine Learning (SVM)**: Structural analysis using HOG (Histogram of Oriented Gradients) features.
3. **Ensemble Voting**: Weighted decision-making mechanism for high-precision diagnosis.

## 🚀 Key Features

*   **Dual-Model Architecture**: Hybrid CNN-SVM ensemble for reduced false positives.
*   **Real-time Analysis**: Instant classification of Kidney CT scans.
*   **User-Friendly Interface**: Interactive web dashboard for easy uploading and result interpretation.
*   **Automated Medical Triage**: Provides preliminary risk assessment and actionable advice.

## 🛠️ Tech Stack

*   **Core**: Python 3.9+
*   **Deep Learning**: TensorFlow, Keras
*   **Computer Vision**: OpenCV, Scikit-Image
*   **ML Algorithms**: Scikit-Learn (SVM)
*   **Interface**: Gradio / Web API

## 📋 Installation & Usage

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python app.py
   ```

## 🧠 Model Architecture

The system utilizes a custom Convolutional Neural Network (CNN) trained on pre-processed CT imagery, running in parallel with a Support Vector Machine (SVM) classifier. The results are aggregated to ensure high sensitivity and specificity in detection.
   ```

3. Launch interface:
   ```bash
   python app.py
   ```

## ⚠️ Medical Disclaimer

**For educational/research purposes only.** 

Not intended for medical diagnosis. Always consult healthcare professionals for proper medical advice.

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.10+
- OpenCV 4.6+
- Gradio 3.50+
- scikit-learn 1.1+

## 📊 Dataset Structure

For training, organize your data as follows:
```
data/
├── Normal/     # Images of healthy kidneys
└── Stone/      # Images containing kidney stones
```

---
*Developed by [Your Name]*
