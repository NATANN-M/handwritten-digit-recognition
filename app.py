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
# Custom CSS for better UI
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
    .confidence-high {
        color: #00cc00;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffaa00;
        font-weight: bold;
    }
    .confidence-low {
        color: #ff4444;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
    }
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
    except:
        st.error("⚠️ Model file not found! Please ensure 'handwritten_digit_cnn.keras' exists.")
        return None

model = load_model()

# --------------------------------------------------
# Improved image preprocessing
# --------------------------------------------------

def preprocess_image(image):
    """
    Convert uploaded image to the format expected by the CNN model.
    """
    # Convert to RGB (in case of RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Convert PIL to numpy array
    image_np = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding for better results with varying lighting
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
        # Fallback: try Otsu thresholding
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
    
    # Add padding around the digit
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
    
    # Resize to 28x28 while maintaining aspect ratio
    digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = digit_resized.astype(np.float32) / 255.0
    
    # Reshape for CNN input (batch_size, height, width, channels)
    normalized = normalized.reshape(1, 28, 28, 1)
    
    return digit_resized, normalized

# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title("🔢 Handwritten Digit Recognition")
st.markdown("Upload an image of a handwritten digit (0-9)")

# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Upload a clear image of a single handwritten digit"
)

# Initialize session state for image rotation
if 'rotated_image' not in st.session_state:
    st.session_state.rotated_image = None
if 'angle' not in st.session_state:
    st.session_state.angle = 0

# --------------------------------------------------
# Process uploaded image
# --------------------------------------------------

if uploaded_file is not None:
    # Load the original image - NO auto-rotation!
    original_image = Image.open(uploaded_file)
    
    # Convert to RGB (this fixes any orientation issues from the file)
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    
    # Store original in session state if not already stored
    if 'original_image' not in st.session_state or st.session_state.original_image is None:
        st.session_state.original_image = original_image.copy()
        st.session_state.rotated_image = original_image.copy()
        st.session_state.angle = 0
    
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
            st.session_state.rotated_image = st.session_state.original_image.copy()
            st.session_state.angle = 0
            st.rerun()
    
    with rot_col4:
        if st.button("📐 Auto Rotate", use_container_width=True):
            # Try to detect the correct orientation
            # This uses a simple heuristic: check if the digit is wider than tall
            img_array = np.array(st.session_state.rotated_image.convert('L'))
            contours, _ = cv2.findContours(
                cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
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
        
        # Show what the model sees for debugging
        with st.expander("🔍 Show processing details"):
            st.write("Try these tips for better results:")
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
            st.caption("What the model sees (28x28 pixels)")
        
        # Make prediction
        if model is not None:
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
                st.session_state.original_image = None
                st.session_state.rotated_image = None
                st.session_state.angle = 0
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
            if st.button("📸 Try Another Image", use_container_width=True):
                st.rerun()

# --------------------------------------------------
# Sidebar information
# --------------------------------------------------

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **Convolutional Neural Network (CNN)** 
    trained on the MNIST dataset to recognize handwritten digits.
    
    **Model:** CNN trained on 60,000 images
    
    **Accuracy:** ~99%
    
    **Input:** 28x28 grayscale images
    """)
    
    st.divider()
    
    st.header("💡 Tips")
    st.markdown("""
    1. Use **clear, well-lit** images
    2. Write **boldly** with good contrast
    3. Center the digit
    4. Use **dark ink on white paper**
    5. Use **Auto Rotate** if the digit is sideways
    6. Avoid multiple digits in one image
    """)
    
    st.divider()
    
    if model is not None:
        st.success("✅ Model ready")
    else:
        st.error("❌ Model not loaded")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption("Made with ❤️ using Streamlit and TensorFlow")