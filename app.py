import os
import io
import base64
import json
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# --------------------------------------------------
# Custom Pure JS Canvas Component Generator
# --------------------------------------------------

COMPONENT_DIR = "custom_canvas"
os.makedirs(COMPONENT_DIR, exist_ok=True)
INDEX_HTML = os.path.join(COMPONENT_DIR, "index.html")

HTML_CODE = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/streamlit-component-lib@1.4.0/dist/streamlit-component-lib.js"></script>
    <style>
        body { 
            margin: 0; 
            padding: 0; 
            background: transparent; 
            font-family: system-ui, -apple-system, sans-serif; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
        }
        #canvas { 
            border: 2px solid #3b82f6; 
            border-radius: 12px; 
            background: #000000; 
            cursor: crosshair; 
            touch-action: none; 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); 
        }
        .controls { 
            margin-top: 12px; 
            display: flex; 
            gap: 10px; 
            width: 280px; 
        }
        .btn-clear { 
            width: 100%; 
            padding: 10px; 
            background: #ef4444; 
            color: white; 
            border: none; 
            border-radius: 8px; 
            font-weight: bold; 
            cursor: pointer; 
            transition: background 0.2s; 
        }
        .btn-clear:hover { background: #dc2626; }
    </style>
</head>
<body>
    <canvas id="canvas" width="280" height="280"></canvas>
    <div class="controls">
        <button class="btn-clear" id="clearBtn">🗑️ Clear Canvas</button>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const clearBtn = document.getElementById('clearBtn');
        let isDrawing = false;

        // Black background, thick white stroke for digit CNN
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 20;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        function sendDataToPython() {
            const dataUrl = canvas.toDataURL('image/png');
            Streamlit.setComponentValue(dataUrl);
        }

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return { x: clientX - rect.left, y: clientY - rect.top };
        }

        function startDrawing(e) {
            isDrawing = true;
            ctx.beginPath();
            const pos = getPos(e);
            ctx.moveTo(pos.x, pos.y);
            e.preventDefault();
        }

        function draw(e) {
            if (!isDrawing) return;
            const pos = getPos(e);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
            e.preventDefault();
        }

        function stopDrawing(e) {
            if (isDrawing) {
                isDrawing = false;
                ctx.closePath();
                sendDataToPython();
            }
        }

        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseleave', stopDrawing);

        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchmove', draw);
        canvas.addEventListener('touchend', stopDrawing);

        clearBtn.addEventListener('click', () => {
            ctx.fillStyle = "#000000";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            Streamlit.setComponentValue(null);
        });

        Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, () => {
            Streamlit.setFrameHeight(340);
        });
    </script>
</body>
</html>
"""

with open(INDEX_HTML, "w", encoding="utf-8") as f:
    f.write(HTML_CODE)

# Register custom component
custom_digit_canvas = components.declare_component("digit_canvas", path=COMPONENT_DIR)

def base64_to_pil(base64_str):
    """Convert canvas Base64 string to PIL Image."""
    if not base64_str or not base64_str.startswith("data:image"):
        return None
    encoded_data = base64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded_data)
    return Image.open(io.BytesIO(image_bytes))

# --------------------------------------------------
# Custom Modern CSS
# --------------------------------------------------

st.markdown("""
    <style>
    .big-digit-card {
        background-color: var(--background-secondary-color, #f8f9fa);
        border: 2px solid var(--border-color, #e0e0e0);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
    }
    .big-digit {
        font-size: 76px;
        font-weight: 800;
        line-height: 1;
        color: var(--text-color, #1f2937);
    }
    .confidence-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 15px;
        margin-top: 10px;
    }
    .conf-high { background-color: #d1fae5; color: #065f46; }
    .conf-med { background-color: #fef3c7; color: #92400e; }
    .conf-low { background-color: #fee2e2; color: #991b1b; }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

TARGET_SIZE = 32

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("handwritten_digit_cnn.keras")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

model = load_model()

# --------------------------------------------------
# Preprocessing Engine
# --------------------------------------------------

def preprocess_image(image, is_inverted=False):
    """
    Preprocess image to 32x32 array for CNN model.
    Handles both photo uploads (dark on light) and canvas drawings (light on dark).
    """
    if image.mode == 'RGBA':
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

def render_prediction_results(processed_img, model_input):
    """Displays prediction visualization, metrics, and distribution chart."""
    if model is None:
        st.warning("Model is not loaded. Cannot run prediction.")
        return

    prediction = model.predict(model_input, verbose=0)[0]
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    col_preview, col_pred = st.columns([1, 1.2])

    with col_preview:
        with st.container(border=True):
            st.markdown("**🔬 Processed Input (32×32)**")
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
                <div style="font-size: 13px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px;">Prediction</div>
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

    # Detailed probabilities expander
    with st.expander("📋 View All Probabilities"):
        cols = st.columns(5)
        for i, prob in enumerate(prediction):
            with cols[i % 5]:
                is_top = (i == predicted_digit)
                st.metric(
                    label=f"Digit {i}" + (" ⭐" if is_top else ""),
                    value=f"{prob*100:.1f}%"
                )

    # Json Download
    result_data = {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": prediction.tolist()
    }
    st.download_button(
        label="📥 Download Result (JSON)",
        data=json.dumps(result_data, indent=2),
        file_name=f"digit_prediction_{predicted_digit}.json",
        mime="application/json",
        use_container_width=True
    )

# --------------------------------------------------
# UI Layout
# --------------------------------------------------

st.title("🔢 Handwritten Digit Classifier")
st.caption("Classify handwritten digits using a 32x32 Convolutional Neural Network")

tab_upload, tab_draw = st.tabs(["📤 Upload Image", "✍️ Draw Digit"])

# --------------------------------------------------
# TAB 1: Image Upload
# --------------------------------------------------

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Upload a clear image of a single digit"
    )

    if uploaded_file is not None:
        # Automatic state refresh on new image upload
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

        st.image(st.session_state.rotated_image, caption="Uploaded Image Preview", use_container_width=True)

        processed_img, model_input = preprocess_image(st.session_state.rotated_image, is_inverted=False)
        if processed_img is None:
            st.error("❌ Could not detect a digit in the image.")
        else:
            st.divider()
            render_prediction_results(processed_img, model_input)

# --------------------------------------------------
# TAB 2: Custom HTML/JS Canvas
# --------------------------------------------------

with tab_draw:
    st.markdown("Draw a single digit (0-9) inside the box below:")
    
    # Trigger native Streamlit JS Component
    canvas_data_url = custom_digit_canvas(key="my_canvas")

    if canvas_data_url:
        canvas_img = base64_to_pil(canvas_data_url)
        if canvas_img is not None:
            gray_canvas = np.array(canvas_img.convert('L'))
            if np.max(gray_canvas) > 20:  # Ensures canvas isn't empty
                processed_img, model_input = preprocess_image(canvas_img, is_inverted=True)
                if processed_img is not None:
                    st.divider()
                    render_prediction_results(processed_img, model_input)
            else:
                st.info("✍️ Draw a digit on the canvas above.")
    else:
        st.info("✍️ Draw a digit on the canvas above.")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    * **Model Architecture:** CNN
    * **Input Dimension:** 32×32 Grayscale
    * **Output:** 10 digit classes (0–9)
    """)
    st.divider()
    if model is not None:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model loading failed")

st.divider()
st.caption("Handwritten Digit Classifier App")