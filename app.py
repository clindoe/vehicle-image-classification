import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for dashboard styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        color: white;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-top: -1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #0f2027, #2c5364);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .prediction-label {
        font-size: 2rem;
        font-weight: 700;
    }
    .confidence-value {
        font-size: 1.3rem;
        color: #4fc3f7;
    }
    .metric-card {
        background: #1e1e2f;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        border: 1px solid #333;
    }
    .metric-card h3 {
        color: #4fc3f7;
        margin: 0;
        font-size: 1.5rem;
    }
    .metric-card p {
        color: #aaa;
        margin: 0;
        font-size: 0.85rem;
    }
    .prob-bar-container {
        margin: 0.4rem 0;
    }
    .prob-label {
        font-size: 0.9rem;
        color: #ccc;
        margin-bottom: 2px;
    }
    .prob-bar-bg {
        background: #1e1e2f;
        border-radius: 8px;
        height: 28px;
        width: 100%;
        position: relative;
        border: 1px solid #333;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        padding-left: 10px;
        font-size: 0.8rem;
        color: white;
        font-weight: 600;
        min-width: 40px;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("vehicle_classifier.keras")
    return model

model = load_model()

# Class names and their icons
class_info = {
    "Auto Rickshaws": "🛺",
    "Bikes": "🚲",
    "Cars": "🚗",
    "Motorcycles": "🏍️",
    "Planes": "✈️",
    "Ships": "🚢",
    "Trains": "🚆"
}
class_names = list(class_info.keys())

# Bar colors for each class
bar_colors = [
    "#ff6b6b", "#ffa502", "#2ed573",
    "#1e90ff", "#a55eea", "#ff4757", "#2bcbba"
]

# Header
st.markdown('<div class="main-header">🚗 Vehicle Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by MobileNetV2 Transfer Learning — 7 Vehicle Categories — 99% Accuracy</div>', unsafe_allow_html=True)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><h3>7</h3><p>Vehicle Classes</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>5,600</h3><p>Training Images</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>99.05%</h3><p>Test Accuracy</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><h3>MobileNetV2</h3><p>Base Model</p></div>', unsafe_allow_html=True)

st.markdown("---")

# Function to make prediction
def predict(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    return predictions[0]

# Function to render probability bars
def render_prob_bars(predictions):
    sorted_indices = np.argsort(predictions)[::-1]
    bars_html = ""
    for i in sorted_indices:
        prob = predictions[i] * 100
        color = bar_colors[i]
        icon = class_info[class_names[i]]
        width = max(prob, 2)
        bars_html += f"""
        <div class="prob-bar-container">
            <div class="prob-label">{icon} {class_names[i]}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: {width}%; background: {color};">
                    {prob:.1f}%
                </div>
            </div>
        </div>
        """
    st.markdown(bars_html, unsafe_allow_html=True)

# Two tabs: Upload or Try Samples
tab1, tab2 = st.tabs(["📤 Upload Image", "🖼️ Try Sample Images"])

with tab1:
    left_col, right_col = st.columns([1, 1])

    with left_col:
        uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with right_col:
        if uploaded_file is not None:
            predictions = predict(image)
            predicted_index = np.argmax(predictions)
            confidence = predictions[predicted_index] * 100
            icon = class_info[class_names[predicted_index]]

            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 3rem;">{icon}</div>
                <div class="prediction-label">{class_names[predicted_index]}</div>
                <div class="confidence-value">{confidence:.2f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Probability Breakdown")
            render_prob_bars(predictions)
        else:
            st.markdown("""
            <div style="text-align: center; color: #888; padding: 3rem;">
                <p style="font-size: 3rem;">📤</p>
                <p>Upload an image to see predictions</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.write("Select a sample image to test the model:")

    sample_dir = "samples"
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if sample_files:
            cols = st.columns(min(len(sample_files), 4))
            for i, sample_file in enumerate(sample_files):
                col_idx = i % 4
                with cols[col_idx]:
                    sample_path = os.path.join(sample_dir, sample_file)
                    img = Image.open(sample_path).convert("RGB")
                    st.image(img, caption=sample_file, use_container_width=True)

                    if st.button(f"Classify", key=f"sample_{i}"):
                        predictions = predict(img)
                        predicted_index = np.argmax(predictions)
                        confidence = predictions[predicted_index] * 100
                        icon = class_info[class_names[predicted_index]]
                        st.success(f"{icon} **{class_names[predicted_index]}** — {confidence:.1f}%")
                        render_prob_bars(predictions)
        else:
            st.info("No sample images found in the 'samples' folder.")
    else:
        st.info("Create a 'samples' folder with a few vehicle images to enable this feature.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.85rem;'>"
    "Built with TensorFlow & Streamlit | Vehicle Classification using MobileNetV2 Transfer Learning"
    "</div>",
    unsafe_allow_html=True
)