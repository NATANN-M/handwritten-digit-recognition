import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
    <style>
    .big-digit {
        font-size: 72px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: #f0f2f6;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-high { color: #00cc00; font-weight: bold; }
    .confidence-medium { color: #ffaa00; font-weight: bold; }
    .confidence-low { color: #ff4444; font-weight: bold; }
    .stButton button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("handwritten_digit_cnn.keras")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

model = load_model()

# Model expects 32x32 input
TARGET_SIZE = 32

# --------------------------------------------------
# Preprocess image for 32x32 model
# --------------------------------------------------

def preprocess_image(image):
    """
    Convert uploaded image to 32x32 format expected by the CNN model.
    """
    # Convert to RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Convert PIL to numpy array
    image_np = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding for better results
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Fallback to Otsu thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, None
    
    # Get the largest contour (the digit)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add small margin to avoid cutting off the digit
    margin = 2
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(thresh.shape[1] - x, w + 2 * margin)
    h = min(thresh.shape[0] - y, h + 2 * margin)
    
    # Crop the digit
    digit = thresh[y:y+h, x:x+w]
    
    # Add padding around the digit (25% of max dimension)
    padding = int(max(w, h) * 0.25)
    digit = cv2.copyMakeBorder(
        digit,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0
    )
    
    # Resize to 32x32 (model expects this size)
    digit_resized = cv2.resize(digit, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = digit_resized.astype(np.float32) / 255.0
    
    # Reshape for CNN input (batch_size, height, width, channels)
    normalized = normalized.reshape(1, TARGET_SIZE, TARGET_SIZE, 1)
    
    return digit_resized, normalized

# --------------------------------------------------
# Reset session state for new image
# --------------------------------------------------

def reset_image_state():
    """Reset all image-related session state variables."""
    st.session_state.original_image = None
    st.session_state.rotated_image = None
    st.session_state.angle = 0
    st.session_state.last_uploaded_file = None

# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title("🔢 Handwritten Digit Recognition")
st.markdown("Upload an image of a handwritten digit (0-9)")

# --------------------------------------------------
# Initialize session state
# --------------------------------------------------

if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'rotated_image' not in st.session_state:
    st.session_state.rotated_image = None
if 'angle' not in st.session_state:
    st.session_state.angle = 0
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Upload a clear image of a single handwritten digit"
)

# --------------------------------------------------
# Handle new upload
# --------------------------------------------------

if uploaded_file is not None:
    # Check if this is a new file (different from last uploaded)
    if st.session_state.last_uploaded_file != uploaded_file.name:
        # New file uploaded - reset everything
        reset_image_state()
        st.session_state.last_uploaded_file = uploaded_file.name
        
        # Load the new image
        original_image = Image.open(uploaded_file)
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        st.session_state.original_image = original_image.copy()
        st.session_state.rotated_image = original_image.copy()
        st.session_state.angle = 0
        
        # Rerun to update the display
        st.rerun()

# --------------------------------------------------
# Process and display image
# --------------------------------------------------

if st.session_state.rotated_image is not None:
    # Display image with rotation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("📷 Your Image")
        st.image(st.session_state.rotated_image, use_container_width=True)
    
    # Rotation controls
    st.subheader("🔄 Adjust Image Rotation")
    
    rot_col1, rot_col2, rot_col3, rot_col4 = st.columns(4)
    
    with rot_col1:
        if st.button("↺ 90° Left", use_container_width=True):
            st.session_state.rotated_image = st.session_state.rotated_image.rotate(90, expand=True)
            st.session_state.angle = (st.session_state.angle + 90) % 360
            st.rerun()
    
    with rot_col2:
        if st.button("↻ 90° Right", use_container_width=True):
            st.session_state.rotated_image = st.session_state.rotated_image.rotate(-90, expand=True)
            st.session_state.angle = (st.session_state.angle - 90) % 360
            st.rerun()
    
    with rot_col3:
        if st.button("🔄 Reset", use_container_width=True):
            if st.session_state.original_image is not None:
                st.session_state.rotated_image = st.session_state.original_image.copy()
                st.session_state.angle = 0
                st.rerun()
    
    with rot_col4:
        if st.button("📐 Auto Rotate", use_container_width=True):
            if st.session_state.rotated_image is not None:
                # Detect if digit is sideways
                img_array = np.array(st.session_state.rotated_image.convert('L'))
                _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # If digit is wider than tall, rotate to make it upright
                    if w > h * 1.2:
                        st.session_state.rotated_image = st.session_state.rotated_image.rotate(-90, expand=True)
                        st.session_state.angle = (st.session_state.angle - 90) % 360
                        st.rerun()
    
    # Show current rotation
    if st.session_state.angle != 0:
        st.caption(f"🔄 Current rotation: {st.session_state.angle}°")
    
    # Process the (potentially rotated) image
    with st.spinner("Processing image..."):
        processed_image, model_input = preprocess_image(st.session_state.rotated_image)
    
    if processed_image is None:
        st.error("❌ Could not detect a digit in the image.")
        st.info("💡 Tips: Use a clear image with good contrast, dark digit on light background.")
        
        with st.expander("🔍 Show processing tips"):
            st.markdown("""
                - Use a **white background** with **dark ink**
                - Center the digit in the image
                - Make sure the digit is **clearly visible**
                - Avoid shadows or reflections
                - Use a **simple font** (not cursive)
                - Try the **Auto Rotate** button if the digit appears sideways
            """)
    else:
        # Show results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✏️ Processed Digit")
            st.image(processed_image, width=200, clamp=True, use_container_width=False)
            st.caption(f"What the model sees (32×32 pixels)")
        
        # Make prediction
        if model is not None:
            try:
                prediction = model.predict(model_input, verbose=0)
                predicted_digit = np.argmax(prediction[0])
                confidence = np.max(prediction[0]) * 100
                
                with col2:
                    st.subheader("🎯 Prediction")
                    
                    # Display confidence with color coding
                    if confidence >= 90:
                        confidence_class = "confidence-high"
                        emoji = "🌟"
                    elif confidence >= 70:
                        confidence_class = "confidence-medium"
                        emoji = "👍"
                    else:
                        confidence_class = "confidence-low"
                        emoji = "🤔"
                    
                    st.markdown(f"""
                        <div class="big-digit">{predicted_digit}</div>
                        <div style="text-align: center; font-size: 20px;">
                            <span class="{confidence_class}">{emoji} {confidence:.1f}% Confidence</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Probability distribution
                st.subheader("📊 Probability Distribution")
                
                # Create DataFrame
                prob_df = pd.DataFrame({
                    'Digit': list(range(10)),
                    'Probability (%)': [p * 100 for p in prediction[0]]
                })
                
                # Display bar chart
                st.bar_chart(prob_df.set_index('Digit'))
                
                # Detailed probabilities in an expander
                with st.expander("📋 View all probabilities"):
                    # Create columns for better display
                    cols = st.columns(5)
                    for i, prob in enumerate(prediction[0]):
                        col_idx = i % 5
                        with cols[col_idx]:
                            is_max = i == predicted_digit
                            st.markdown(f"""
                                <div style="padding: 5px; background: {'#e6f3ff' if is_max else 'transparent'}; 
                                            border-radius: 5px; text-align: center;">
                                    <strong>{'⭐ ' if is_max else ''}{i}</strong>
                                    <br>{prob*100:.1f}%
                                </div>
                            """, unsafe_allow_html=True)
                            st.progress(float(prob))
                
                # Quick actions
                st.divider()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Process Another", use_container_width=True):
                        reset_image_state()
                        st.rerun()
                
                with col2:
                    # Create a downloadable result
                    result_json = {
                        "predicted_digit": int(predicted_digit),
                        "confidence": float(confidence),
                        "probabilities": [float(p) for p in prediction[0]]
                    }
                    st.download_button(
                        label="📥 Download Results",
                        data=str(result_json),
                        file_name=f"digit_prediction_{predicted_digit}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    if st.button("🔄 New Image", use_container_width=True):
                        reset_image_state()
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("Please check if the model is properly loaded and expects 32x32 input.")

# --------------------------------------------------
# Show instructions when no image is loaded
# --------------------------------------------------

else:
    st.info("👆 Upload an image to get started!")
    
    # Show example of what to upload
    with st.expander("📸 What kind of image should I upload?"):
        st.markdown("""
            **Best practices:**
            - ✍️ Handwritten digit (0-9) on **white paper**
            - 🖊️ Use **dark ink** (black or dark blue)
            - 📏 Digit should be **centered** in the image
            - 💡 Good **lighting** with no shadows
            - 📐 Image can be any size (app will resize)
            - 🚫 Avoid multiple digits or text
            - 🖼️ Use **simple fonts** (not cursive)
            
            **File formats supported:** JPG, JPEG, PNG, BMP, TIFF
        """)

# --------------------------------------------------
# Sidebar information
# --------------------------------------------------

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **Convolutional Neural Network (CNN)** 
    trained on handwritten digits.
    
    **Model Input:** 32×32 grayscale images
    
    **Output:** 10 digits (0-9)
    
    **Accuracy:** High accuracy on MNIST-like digits
    """)
    
    st.divider()
    
    st.header("💡 Tips for Best Results")
    st.markdown("""
    1. 📸 Use **clear, well-lit** images
    2. ✍️ Write **boldly** with good contrast
    3. 🎯 Center the digit in the frame
    4. ⚫ Use **dark ink on white paper**
    5. 🔄 Use **Auto Rotate** if sideways
    6. ❌ Avoid multiple digits
    7. 📏 Keep the digit proportional
    """)
    
    st.divider()
    
    if model is not None:
        st.success("✅ Model loaded successfully!")
        st.info(f"📐 Input size: {TARGET_SIZE}×{TARGET_SIZE} pixels")
    else:
        st.error("❌ Model not loaded")
    
    st.divider()
    
    # Show current file info
    if st.session_state.last_uploaded_file:
        st.caption(f"📎 Current: {st.session_state.last_uploaded_file}")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
