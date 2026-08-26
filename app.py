import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import io

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS for better UI
# --------------------------------------------------

st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 10px 0;
    }
    .digit-display {
        font-size: 80px;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("handwritten_digit_cnn.keras")
    except:
        st.error("⚠️ Model file not found! Please ensure 'handwritten_digit_cnn.keras' exists.")
        return None

model = load_model()

# --------------------------------------------------
# Preprocess image (IMPROVED)
# --------------------------------------------------

def preprocess_image(image, auto_crop=True, padding_ratio=0.25):
    """
    Convert uploaded image into the same format used during CNN training.
    """
    # Convert PIL image to RGB
    image = image.convert("RGB")
    
    # Convert PIL -> NumPy
    image_np = np.array(image)
    
    # RGB -> grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Improve contrast using CLAHE (better than simple normalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding (better for different lighting conditions)
    thresh = cv2.adaptiveThreshold(
        gray_blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # Optionally, use Otsu as fallback
    if np.count_nonzero(thresh) < 50:  # If adaptive threshold fails
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if not auto_crop:
        # Just resize the whole image
        processed = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
        processed = processed.astype(np.float32) / 255.0
        processed = processed.reshape(1, 28, 28, 1)
        return thresh, processed
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None
    
    # Find largest contour
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add small margin if contour touches border
    margin = 2
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(thresh.shape[1] - x, w + 2 * margin)
    h = min(thresh.shape[0] - y, h + 2 * margin)
    
    # Crop digit
    digit = thresh[y:y+h, x:x+w]
    
    # Add padding
    padding = int(max(w, h) * padding_ratio)
    digit = cv2.copyMakeBorder(
        digit,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=0
    )
    
    # Resize while keeping proportions
    h2, w2 = digit.shape
    scale = 28 / max(h2, w2)
    new_w = max(1, int(w2 * scale))
    new_h = max(1, int(h2 * scale))
    
    digit_resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )
    
    # Put in 32x32 canvas
    canvas = np.zeros((32, 32), dtype=np.uint8)
    start_x = (32 - new_w) // 2
    start_y = (32 - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = digit_resized
    
    # Normalize to [0, 1] and reshape for CNN
    normalized = canvas.astype(np.float32) / 255.0
    normalized = normalized.reshape(1, 32, 32, 1)
    
    return canvas, normalized

# --------------------------------------------------
# Plot probability distribution
# --------------------------------------------------

def plot_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(8, 4))
    digits = range(10)
    colors = ['#1f77b4' if i != np.argmax(probabilities) else '#ff7f0e' for i in digits]
    bars = ax.bar(digits, probabilities * 100, color=colors, alpha=0.7)
    
    # Add value labels on top of bars
    for bar, prob in zip(bars, probabilities * 100):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Digit')
    ax.set_ylabel('Probability (%)')
    ax.set_title('Prediction Confidence Distribution')
    ax.set_xticks(digits)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at 10% (random chance)
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Random chance')
    ax.legend()
    
    return fig

# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title("🔢 Handwritten Digit Recognition")
st.markdown("### Upload or draw a handwritten digit (0-9)")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🎨 Draw Digit", "📸 Camera"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload an image containing a handwritten digit"
    )

with tab2:
    st.info("🎨 Drawing canvas coming soon! For now, use the upload or camera option.")

with tab3:
    st.info("📸 Camera input coming soon! For now, use the upload option.")

# --------------------------------------------------
# Process uploaded image
# --------------------------------------------------

if uploaded_file is not None:
    # Display filename and info
    st.caption(f"📎 File: {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
    
    # Open image
    image = Image.open(uploaded_file)
    
    # IMPORTANT FIX: Don't auto-rotate based on EXIF
    # Just convert to RGB and keep orientation as-is
    image = image.convert("RGB")
    
    # Display original image (without rotation)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("📷 Original Image")
        st.image(image, width=300, use_container_width=True)
    
    # Add rotation controls
    st.subheader("🔄 Image Controls")
    rotation_col1, rotation_col2, rotation_col3 = st.columns(3)
    
    with rotation_col1:
        if st.button("↺ Rotate Left"):
            image = image.rotate(90, expand=True)
            st.rerun()
    
    with rotation_col2:
        if st.button("↻ Rotate Right"):
            image = image.rotate(-90, expand=True)
            st.rerun()
    
    with rotation_col3:
        if st.button("🔄 Reset"):
            image = Image.open(uploaded_file).convert("RGB")
            st.rerun()
    
    # Preprocess with options
    st.subheader("⚙️ Preprocessing Options")
    col1, col2 = st.columns(2)
    
    with col1:
        auto_crop = st.checkbox("Auto-crop digit", value=True)
    
    with col2:
        padding_ratio = st.slider("Padding ratio", 0.1, 0.5, 0.25, 0.05)
    
    # Process image
    with st.spinner("Processing image..."):
        processed_image, model_input = preprocess_image(image, auto_crop, padding_ratio)
    
    if processed_image is None:
        st.error("❌ Could not detect a digit in the image. Please try with a clearer image.")
        
        # Show the processed threshold image for debugging
        st.subheader("🔍 What the model sees")
        gray = np.array(image.convert('L'))
        _, thresh_display = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        st.image(thresh_display, width=200, clamp=True)
        st.caption("This is the thresholded image. If you don't see a clear digit, try a different image.")
        
    else:
        # Show processed image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✏️ Processed Digit")
            st.image(processed_image, width=200, clamp=True, use_container_width=False)
            st.caption(f"Processed to {processed_image.shape[0]}×{processed_image.shape[1]} pixels")
        
        # Prediction
        if model is not None:
            prediction = model.predict(model_input, verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100
            
            with col2:
                st.subheader("🎯 Prediction")
                
                # Big digit display
                st.markdown(f"""
                    <div class="prediction-box">
                        <div>Predicted Digit:</div>
                        <div class="digit-display">{predicted_digit}</div>
                        <div style="font-size: 24px; color: {'#4CAF50' if confidence > 80 else '#FFA500' if confidence > 50 else '#FF4444'}">
                            {confidence:.1f}% Confidence
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Show probability distribution
        st.subheader("📊 Probability Distribution")
        fig = plot_probabilities(prediction[0])
        st.pyplot(fig)
        plt.close(fig)
        
        # Detailed probability table
        with st.expander("📋 Detailed Probabilities"):
            st.write("| Digit | Probability | Bar |")
            st.write("|-------|------------|-----|")
            max_prob = np.max(prediction[0])
            for i, prob in enumerate(prediction[0]):
                bar_length = int(prob * 100)
                bar = "█" * min(bar_length, 50)
                emoji = "⭐" if i == predicted_digit else "  "
                st.write(f"| {i} {emoji} | {prob*100:.2f}% | {bar} |")
        
        # Save result button
        st.download_button(
            label="📥 Download Result",
            data=str({
                "predicted_digit": int(predicted_digit),
                "confidence": float(confidence),
                "probabilities": [float(p) for p in prediction[0]]
            }),
            file_name=f"prediction_digit_{predicted_digit}.json",
            mime="application/json"
        )
        
        # Tips for better results
        with st.expander("💡 Tips for Better Recognition"):
            st.markdown("""
                - Use **clear, well-lit** images
                - Write the digit **boldly** with high contrast
                - Center the digit in the image
                - Avoid background noise or text
                - Use **dark ink on white paper** for best results
                - Try different padding ratios if the digit is too small or large
            """)

# --------------------------------------------------
# Sidebar with information
# --------------------------------------------------

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **Convolutional Neural Network (CNN)** 
    trained on the MNIST dataset to recognize handwritten digits.
    
    **Model Architecture:**
    - CNN with 3 convolutional layers
    - Max pooling and dropout
    - Dense layers with ReLU activation
    - Softmax output for 10 classes
    
    **Dataset:** MNIST (60,000 training images)
    
    **Accuracy:** ~99%
    """)
    
    st.divider()
    
    st.header("📌 Instructions")
    st.markdown("""
    1. Upload an image with a digit
    2. Adjust rotation if needed
    3. View the processed digit
    4. See the prediction results
    5. Download results if needed
    """)
    
    st.divider()
    
    st.header("🔧 Requirements")
    st.code("""
    streamlit>=1.28.0
    tensorflow>=2.13.0
    opencv-python
    numpy
    pillow
    matplotlib
    """, language="text")
    
    st.divider()
    
    if model is not None:
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Model not loaded")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption("Made with ❤️ using Streamlit and TensorFlow")