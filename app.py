import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
import pandas as pd
import json

# Try importing drawable canvas, handle gracefully if not installed
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Handwritten Digit AI",
    page_icon="🔢",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS for Modern UI
# --------------------------------------------------

st.markdown("""
    <style>
    .big-digit-card {
        background-color: var(--background-secondary-color, #f8f9fa);
        border: 2px solid var(--border-color, #e0e0e0);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 15px;
    }
    .big-digit {
        font-size: 80px;
        font-weight: 800;
        line-height: 1;
        color: var(--text-color, #1f2937);
    }
    .confidence-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 16px;
        margin-top: 10px;
    }
    .conf-high { background-color: #d1fae5; color: #065f46; }
    .conf-med { background-color: #fef3c7; color: #92400e; }
    .conf-low { background-color: #fee2e2; color: #991b1b; }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load model
# --------------------------------------------------

TARGET_SIZE = 32

@st.cache_resource
def load_digit_model():
    try:
        model = tf.keras.models.load_model("handwritten_digit_cnn.keras")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model file (`handwritten_digit_cnn.keras`): {str(e)}")
        return None

model = load_digit_model()

# --------------------------------------------------
# Preprocessing Engine
# --------------------------------------------------

def preprocess_image(image, is_inverted=False):
    """
    Convert image to 32x32 format expected by the CNN model.
    Handles both dark-on-light and light-on-dark images.
    """
    if image.mode == 'RGBA':
        # Create white background for transparent PNGs
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    if is_inverted:
        thresh = gray
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if not is_inverted:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    
    margin = 4
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(thresh.shape[1] - x, w + 2 * margin)
    h = min(thresh.shape[0] - y, h + 2 * margin)
    
    digit = thresh[y:y+h, x:x+w]
    padding = int(max(w, h) * 0.25)
    digit = cv2.copyMakeBorder(
        digit, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=0
    )
    
    digit_resized = cv2.resize(digit, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    normalized = digit_resized.astype(np.float32) / 255.0
    normalized = normalized.reshape(1, TARGET_SIZE, TARGET_SIZE, 1)
    
    return digit_resized, normalized

def display_prediction_results(processed_img, model_input):
    """Renders the prediction results UI card & charts."""
    if model is None:
        st.warning("Model is not loaded. Cannot generate predictions.")
        return

    prediction = model.predict(model_input, verbose=0)[0]
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    col_preview, col_pred = st.columns([1, 1.2])

    with col_preview:
        with st.container(border=True):
            st.markdown("**🔬 Input View (32×32)**")
            st.image(processed_img, use_container_width=True)

    with col_pred:
        if confidence >= 90:
            badge_class, emoji = "conf-high", "🌟 High Confidence"
        elif confidence >= 70:
            badge_class, emoji = "conf-med", "👍 Moderate Confidence"
        else:
            badge_class, emoji = "conf-low", "🤔 Low Confidence"

        st.markdown(f"""
            <div class="big-digit-card">
                <div style="font-size: 14px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px;">Predicted Digit</div>
                <div class="big-digit">{predicted_digit}</div>
                <div class="confidence-badge {badge_class}">{emoji} ({confidence:.1f}%)</div>
            </div>
        """, unsafe_allow_html=True)

    # Probability Distribution Chart
    st.subheader("📊 Class Probabilities")
    prob_df = pd.DataFrame({
        'Digit': [str(i) for i in range(10)],
        'Probability (%)': prediction * 100
    }).set_index('Digit')
    
    st.bar_chart(prob_df, height=220)

    # Detailed breakdown expander
    with st.expander("📋 Detailed Breakdown"):
        cols = st.columns(5)
        for i, prob in enumerate(prediction):
            with cols[i % 5]:
                is_top = (i == predicted_digit)
                st.metric(
                    label=f"Digit {i}" + (" ⭐" if is_top else ""),
                    value=f"{prob*100:.1f}%"
                )

    # Download result
    result_data = {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": prediction.tolist()
    }
    st.download_button(
        label="📥 Download Prediction Result (JSON)",
        data=json.dumps(result_data, indent=2),
        file_name=f"digit_prediction_{predicted_digit}.json",
        mime="application/json",
        use_container_width=True
    )

# --------------------------------------------------
# Header & UI Tabs
# --------------------------------------------------

st.title("🔢 Handwritten Digit Classifier")
st.caption("Classify digits (0–9) using a 32x32 Convolutional Neural Network")

tab_upload, tab_draw = st.tabs(["📤 Upload Image", "✍️ Draw Digit"])

# --------------------------------------------------
# TAB 1: Upload Image Mode
# --------------------------------------------------

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a clear picture of a single digit on a clean background."
    )

    if uploaded_file is not None:
        # Detect new image upload automatically & update state
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get('current_file_key') != file_key:
            st.session_state.current_file_key = file_key
            img = Image.open(uploaded_file)
            st.session_state.original_image = img
            st.session_state.rotated_image = img.copy()
            st.session_state.angle = 0

        # Image adjustment controls
        with st.expander("🔄 Image Controls & Rotation", expanded=False):
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                if st.button("↺ 90° Left", use_container_width=True):
                    st.session_state.rotated_image = st.session_state.rotated_image.rotate(90, expand=True)
                    st.session_state.angle = (st.session_state.angle + 90) % 360
                    st.rerun()
            with r_col2:
                if st.button("↻ 90° Right", use_container_width=True):
                    st.session_state.rotated_image = st.session_state.rotated_image.rotate(-90, expand=True)
                    st.session_state.angle = (st.session_state.angle - 90) % 360
                    st.rerun()
            with r_col3:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.rotated_image = st.session_state.original_image.copy()
                    st.session_state.angle = 0
                    st.rerun()
            with r_col4:
                if st.button("📐 Auto-Orient", use_container_width=True):
                    img_gray = np.array(st.session_state.rotated_image.convert('L'))
                    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        _, _, w, h = cv2.boundingRect(cnt)
                        if w > h * 1.2:
                            st.session_state.rotated_image = st.session_state.rotated_image.rotate(-90, expand=True)
                            st.session_state.angle = (st.session_state.angle - 90) % 360
                            st.rerun()

        # Display image preview
        st.image(st.session_state.rotated_image, caption="Uploaded Image Preview", use_container_width=True)

        # Process image and display prediction
        processed_img, model_input = preprocess_image(st.session_state.rotated_image)
        if processed_img is None:
            st.error("❌ Could not detect a clear digit contour in the image.")
        else:
            st.divider()
            display_prediction_results(processed_img, model_input)

# --------------------------------------------------
# TAB 2: Draw Canvas Mode
# --------------------------------------------------

with tab_draw:
    if not CANVAS_AVAILABLE:
        st.info("💡 To enable live canvas drawing, install `streamlit-drawable-canvas`:")
        st.code("pip install streamlit-drawable-canvas", language="bash")
    else:
        st.markdown("Draw a single digit (0-9) inside the box below:")
        
        canvas_col, _ = st.columns([1, 0.01])
        with canvas_col:
            canvas_result = st_canvas(
                fill_color="#000000",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="digit_canvas",
            )

        if canvas_result.image_data is not None:
            # Extract RGB channels from canvas
            canvas_img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
            
            # Check if user has drawn anything (non-black pixels)
            gray_canvas = np.array(canvas_img.convert('L'))
            if np.max(gray_canvas) > 20:  # Has drawing activity
                processed_img, model_input = preprocess_image(canvas_img, is_inverted=True)
                if processed_img is not None:
                    st.divider()
                    display_prediction_results(processed_img, model_input)
            else:
                st.info("✍️ Draw a digit above to see real-time predictions.")

# --------------------------------------------------
# Sidebar Information
# --------------------------------------------------

with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown("""
    * **Architecture:** Convolutional Neural Network (CNN)
    * **Input Dimension:** 32×32 Grayscale
    * **Target Classes:** 10 classes (Digits 0 through 9)
    """)
    st.divider()
    if model is not None:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model load failed")

# Sidebar styling cleanups
st.divider()
st.caption("Handwritten Digit Classifier App")