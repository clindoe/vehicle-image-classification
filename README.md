cat > README.md << 'EOF'
# 🚗 Vehicle Image Classification

A deep learning web application that classifies vehicle images into **7 categories** using **MobileNetV2 transfer learning**, achieving **99.05% test accuracy**.

## 🔗 Live Demo

**[Try the App →](https://vehicle-image-classification-j7.streamlit.app/)**

## 📸 Categories

| Category | Images |
|----------|--------|
| 🛺 Auto Rickshaws | 800 |
| 🚲 Bikes | 800 |
| 🚗 Cars | 790 |
| 🏍️ Motorcycles | 800 |
| ✈️ Planes | 800 |
| 🚢 Ships | 800 |
| 🚆 Trains | 800 |

## 🏗️ Architecture

- **Base Model:** MobileNetV2 (pretrained on ImageNet, 2.2M frozen parameters)
- **Custom Layers:** GlobalAveragePooling2D → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.3) → Dense(7, Softmax)
- **Trainable Parameters:** 164,871 (out of 2.4M total)
- **Dataset:** [Kaggle Vehicle Classification](https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification) — 5,600 images
- **Data Split:** 70% Train / 15% Validation / 15% Test

## 📊 Results

| Metric | Score |
|--------|-------|
| Training Accuracy | 99.44% |
| Validation Accuracy | 98.44% |
| **Test Accuracy** | **99.05%** |

## 🛠️ Tech Stack

- **TensorFlow / Keras** — Model training & transfer learning
- **MobileNetV2** — Pretrained convolutional neural network
- **Streamlit** — Interactive web UI
- **NumPy & Pillow** — Image preprocessing
- **scikit-learn & Matplotlib** — Evaluation & visualization

## 🚀 Run Locally
```bash
git clone https://github.com/clindoe/vehicle-image-classification.git
cd vehicle-image-classification
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
```
├── app.py                       # Streamlit web application
├── vehicle_classifier.keras     # Trained model weights
├── vehicle_classifier.ipynb     # Training notebook
├── samples/                     # Sample images for testing
├── requirements.txt             # Python dependencies
└── README.md
```

## 💡 Key Concepts

- **Transfer Learning:** Leveraged MobileNetV2's pretrained ImageNet weights instead of training from scratch, enabling high accuracy with only 5,600 images
- **Data Augmentation Prevention:** Used Dropout layers (30%) to prevent overfitting on the relatively small dataset
- **Efficient Inference:** Model preprocessing (rescaling) is built into the network architecture for seamless deployment
EOF

