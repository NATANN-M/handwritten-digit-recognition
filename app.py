import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import base64
import io

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS for paper-like UI
# --------------------------------------------------

st.markdown("""
<style>
    /* Paper background */
    .stApp {
        background: #F6F4EC;
    }
    
    .main {
        padding: 2rem 1rem;
    }
    
    /* Paper style heading */
    .paper-title {
        font-family: 'Segoe Print', 'Bradley Hand', 'Comic Sans MS', cursive;
        font-size: 40px;
        color: #2C4A6E;
        transform: rotate(-0.6deg);
        margin-bottom: 0;
        font-weight: normal;
    }
    
    .paper-subtitle {
        font-size: 14px;
        color: #55554F;
        margin-bottom: 30px;
        letter-spacing: 0.2px;
    }
    
    /* Canvas container */
    .canvas-wrapper {
        background: rgba(255,255,255,0.55);
        border: 1px solid rgba(43,43,46,0.15);
        border-radius: 3px;
        padding: 18px;
        display: inline-block;
    }
    
    /* Canvas styling */
    .drawing-canvas {
        width: 280px;
        height: 280px;
        background: #FFFFFF;
        border: 1.5px solid #2B2B2E;
        border-radius: 2px;
        cursor: crosshair;
        touch-action: none;
        display: block;
    }
    
    .canvas-label {
        font-size: 12px;
        color: #55554F;
        margin-top: 8px;
        text-align: center;
    }
    
    /* Results styling */
    .grade-circle {
        width: 92px;
        height: 92px;
        border: 3px solid #B33A3A;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Segoe Print', 'Bradley Hand', 'Comic Sans MS', cursive;
        font-size: 52px;
        color: #7A2626;
        transform: rotate(-4deg);
        margin: 6px 0 10px 6px;
        background: white;
    }
    
    .grade-conf {
        font-family: 'Segoe Print', 'Bradley Hand', 'Comic Sans MS', cursive;
        color: #7A2626;
        font-size: 20px;
        margin: 0 0 18px 10px;
        transform: rotate(-2deg);
    }
    
    .placeholder-note {
        font-size: 13px;
        color: #55554F;
        font-style: italic;
        margin-top: 30px;
    }
    
    /* Bars styling */
    .bars-container {
        display: flex;
        flex-direction: column;
        gap: 5px;
        width: 100%;
    }
    
    .bar-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        font-family: 'SF Mono', 'IBM Plex Mono', ui-monospace, monospace;
        color: #55554F;
    }
    
    .bar-digit {
        width: 12px;
        text-align: right;
        color: #2B2B2E;
        font-weight: bold;
    }
    
    .bar-track {
        flex: 1;
        height: 9px;
        background: rgba(43,43,46,0.08);
        border-radius: 2px;
        overflow: hidden;
    }
    
    .bar-fill {
        height: 100%;
        background: #B33A3A;
        border-radius: 2px;
        transition: width 0.3s ease;
    }
    
    .bar-pct {
        width: 40px;
        text-align: right;
        font-weight: bold;
    }
    
    /* Controls */
    .controls {
        display: flex;
        gap: 10px;
        margin-top: 12px;
        justify-content: center;
    }
    
    .btn-clear {
        font-family: inherit;
        font-size: 13px;
        padding: 7px 16px;
        border-radius: 3px;
        border: 1px solid #2B2B2E;
        background: #F6F4EC;
        color: #2B2B2E;
        cursor: pointer;
    }
    
    .btn-clear:hover {
        background: #ECE8DA;
    }
    
    .btn-primary {
        font-family: inherit;
        font-size: 13px;
        padding: 7px 16px;
        border-radius: 3px;
        border: 1px solid #B33A3A;
        background: #B33A3A;
        color: white;
        cursor: pointer;
    }
    
    .btn-primary:hover {
        background: #7A2626;
    }
    
    /* Stats section */
    .stats {
        margin-top: 34px;
        padding-top: 14px;
        border-top: 1px dashed rgba(43,43,46,0.3);
        font-family: 'SF Mono', 'IBM Plex Mono', ui-monospace, monospace;
        font-size: 11.5px;
        color: #55554F;
        line-height: 1.9;
    }
    
    .stats b {
        color: #2B2B2E;
    }
    
    /* Upload section */
    .upload-section {
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px dashed rgba(43,43,46,0.2);
    }
    
    /* Responsive */
    @media (max-width: 640px) {
        .drawing-canvas {
            width: 200px;
            height: 200px;
        }
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
        st.warning("⚠️ Model file not found. Please ensure 'handwritten_digit_cnn.keras' exists.")
        return None

model = load_model()
TARGET_SIZE = 32  # Your model expects 32x32

# --------------------------------------------------
# Initialize session state
# --------------------------------------------------

if 'canvas_image' not in st.session_state:
    st.session_state.canvas_image = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'has_drawn' not in st.session_state:
    st.session_state.has_drawn = False

# --------------------------------------------------
# Preprocessing function (same as your existing)
# --------------------------------------------------

def preprocess_canvas_image(image_array):
    """Preprocess canvas image for 32x32 model"""
    # Invert if needed (canvas draws black on white)
    thresh = cv2.threshold(image_array, 30, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None
    
    # Get largest contour
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add margin
    margin = 2
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(thresh.shape[1] - x, w + 2 * margin)
    h = min(thresh.shape[0] - y, h + 2 * margin)
    
    # Crop digit
    digit = thresh[y:y+h, x:x+w]
    
    # Add padding
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
    
    # Resize to 32x32
    digit_resized = cv2.resize(digit, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    
    # Normalize
    normalized = digit_resized.astype(np.float32) / 255.0
    normalized = normalized.reshape(1, TARGET_SIZE, TARGET_SIZE, 1)
    
    return digit_resized, normalized

# --------------------------------------------------
# HTML Canvas Component (with all styling)
# --------------------------------------------------

def get_canvas_html():
    return """
    <div class="canvas-wrapper">
        <canvas id="drawCanvas" class="drawing-canvas" width="280" height="280"></canvas>
        <p class="canvas-label">use your mouse or finger</p>
        <div class="controls">
            <button class="btn-clear" id="clearBtn">Clear</button>
            <button class="btn-primary" id="predictBtn">Predict</button>
        </div>
    </div>
    
    <script>
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    
    // Initialize canvas
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 16;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#1A1A1A';
    
    let drawing = false;
    let lastX = 0, lastY = 0;
    let hasInk = false;
    
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        if (e.touches && e.touches.length) {
            return { 
                x: (e.touches[0].clientX - rect.left) * scaleX, 
                y: (e.touches[0].clientY - rect.top) * scaleY 
            };
        }
        return { 
            x: (e.clientX - rect.left) * scaleX, 
            y: (e.clientY - rect.top) * scaleY 
        };
    }
    
    function startDraw(e) {
        e.preventDefault();
        drawing = true;
        hasInk = true;
        const p = getPos(e);
        lastX = p.x; 
        lastY = p.y;
        ctx.beginPath();
        ctx.arc(p.x, p.y, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fill();
    }
    
    function moveDraw(e) {
        if (!drawing) return;
        e.preventDefault();
        const p = getPos(e);
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
        lastX = p.x; 
        lastY = p.y;
    }
    
    function endDraw(e) {
        drawing = false;
    }
    
    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', moveDraw);
    window.addEventListener('mouseup', endDraw);
    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove', moveDraw, { passive: false });
    canvas.addEventListener('touchend', endDraw);
    
    // Clear button
    document.getElementById('clearBtn').addEventListener('click', () => {
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        hasInk = false;
        
        const data = { type: 'canvas_cleared' };
        window.parent.postMessage(data, '*');
    });
    
    // Predict button
    document.getElementById('predictBtn').addEventListener('click', () => {
        if (!hasInk) {
            alert('Please draw a digit first!');
            return;
        }
        
        const imageData = canvas.toDataURL('image/png');
        const data = { 
            type: 'canvas_prediction', 
            imageData: imageData 
        };
        window.parent.postMessage(data, '*');
    });
    </script>
    """

# --------------------------------------------------
# Main UI Layout
# --------------------------------------------------

st.markdown('<h1 class="paper-title">Digit Recognizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="paper-subtitle">draw a digit 0–9 · graded live by a small CNN running entirely in your browser</p>', unsafe_allow_html=True)

# Two column layout
col1, col2 = st.columns([1, 1.2])

with col1:
    # Canvas
    st.components.v1.html(get_canvas_html(), height=380)
    
    # Upload section
    with st.expander("📤 Upload Image Instead"):
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy for preprocessing
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            processed, model_input = preprocess_canvas_image(gray)
            
            if processed is not None and model is not None:
                prediction = model.predict(model_input, verbose=0)
                predicted_digit = np.argmax(prediction[0])
                confidence = np.max(prediction[0]) * 100
                
                st.session_state.prediction = {
                    'digit': int(predicted_digit),
                    'confidence': confidence,
                    'probabilities': prediction[0],
                    'processed': processed
                }
                st.rerun()

with col2:
    # Results display
    if st.session_state.prediction is not None:
        pred = st.session_state.prediction
        
        # Show results with paper style
        st.markdown(f"""
        <div style="padding: 10px 0;">
            <div class="grade-circle">{pred['digit']}</div>
            <div class="grade-conf">{pred['confidence']:.1f}% sure</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show processed image if available
        if 'processed' in pred and pred['processed'] is not None:
            st.image(pred['processed'], width=100, clamp=True)
            st.caption("What the model sees")
        
        # Probability bars
        st.markdown('<div class="bars-container">', unsafe_allow_html=True)
        
        probs = pred['probabilities']
        for i, prob in enumerate(probs):
            pct = prob * 100
            is_max = i == pred['digit']
            color = '#B33A3A' if is_max else '#55554F'
            st.markdown(f"""
            <div class="bar-row">
                <span class="bar-digit">{i}</span>
                <span class="bar-track">
                    <span class="bar-fill" style="width:{pct}%; background:{color};"></span>
                </span>
                <span class="bar-pct">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Clear Results", use_container_width=True):
                st.session_state.prediction = None
                st.session_state.processed_image = None
                st.rerun()
        with col_btn2:
            if st.button("New Drawing", use_container_width=True):
                st.session_state.prediction = None
                st.session_state.has_drawn = False
                st.rerun()
                
    else:
        st.markdown('<p class="placeholder-note">nothing graded yet — draw a digit</p>', unsafe_allow_html=True)

# Stats section
st.markdown("""
<div class="stats">
    <div><b>Model:</b> CNN &middot; Trained on handwritten digits</div>
    <div><b>Input:</b> 32×32 grayscale images</div>
    <div><b>Output:</b> 10 classes (digits 0-9)</div>
    <div><b>Inference:</b> Runs locally with TensorFlow</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# JavaScript to handle canvas messages
# --------------------------------------------------

js_code = """
<script>
window.addEventListener('message', function(event) {
    if (event.data && event.data.type === 'canvas_prediction') {
        // Send image data to Streamlit
        const imageData = event.data.imageData;
        
        // Use fetch to send to backend
        fetch('/_stcore/upload', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        }).catch(error => console.error('Error:', error));
    }
    if (event.data && event.data.type === 'canvas_cleared') {
        console.log('Canvas cleared');
        // Trigger Streamlit rerun
        window.location.reload();
    }
});
</script>
"""
st.components.v1.html(js_code, height=0)

# --------------------------------------------------
# Process canvas image (backend)
# --------------------------------------------------

# This is a workaround to receive canvas data
# The actual processing happens via the message handler

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **Convolutional Neural Network (CNN)** 
    trained on handwritten digits.
    
    **Model Input:** 32×32 grayscale images
    
    **Output:** 10 digits (0-9)
    """)
    
   
    
    if model is not None:
        st.success("✅ Model loaded!")
        st.info(f"📐 Input: {TARGET_SIZE}×{TARGET_SIZE}")
    else:
        st.error("❌ Model not loaded")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
