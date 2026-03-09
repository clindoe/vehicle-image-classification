# Vehicle Image Classification

A deep learning web app that classifies vehicle images into 7 categories using MobileNetV2 transfer learning.

## Categories
Auto Rickshaws, Bikes, Cars, Motorcycles, Planes, Ships, Trains

## Model Details
- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Dataset:** 5,600 images from Kaggle
- **Test Accuracy:** 99.05%
- **Framework:** TensorFlow / Keras

## Tech Stack
- TensorFlow / Keras (model training)
- MobileNetV2 (transfer learning)
- Streamlit (web UI)
- Streamlit Cloud (deployment)

## How to Run Locally
1. Clone this repo
2. Install dependencies: pip install -r requirements.txt
3. Run: streamlit run app.py
