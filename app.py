import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import base64
from io import BytesIO
import time

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS for drawing canvas
# --------------------------------------------------

st.markdown("""
    <style>
    .stApp {
        background: #F6F4EC;
    }
    .canvas-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        background: rgba(255,255,255,0.5);
        border-radius: 12px;
        border: 1px solid rgba(43,43,46,0.15);
    }
    .drawing-canvas {
        border: 2px solid #2B2B2E;
        border-radius: 8px;
        background: #FFFFFF;
        cursor: crosshair;
        touch-action: none;
        width: 280px;
        height: 280px;
    }
    .prediction-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: rgba(255,255,255,0.8);
        border-radius: 10px;
        border: 1px solid rgba(43,43,46,0.15);
        min-height: 200px;
    }
    .digit-display {
        font-size: 72px;
        font-weight: bold;
        color: #2B2B2E;
        margin: 10px 0;
        font-family: 'Segoe Print', 'Bradley Hand', cursive;
    }
    .confidence-text {
        font-size: 20px;
        color: #B33A3A;
        font-family: 'Segoe Print', 'Bradley Hand', cursive;
    }
    .bar-container {
        width: 100%;
        margin: 4px 0;
    }
    .bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        font-family: 'SF Mono', monospace;
        color: #555;
    }
    .bar-track {
        width: 100%;
        height: 8px;
        background: rgba(43,43,46,0.08);
        border-radius: 4px;
        overflow: hidden;
        margin: 2px 0;
    }
    .bar-fill {
        height: 100%;
        background: #B33A3A;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .controls {
        display: flex;
        gap: 10px;
        margin-top: 12px;
    }
    .btn-clear {
        padding: 8px 24px;
        border: 1px solid #2B2B2E;
        background: #F6F4EC;
        border-radius: 4px;
        cursor: pointer;
        font-family: inherit;
        font-size: 13px;
    }
    .btn-predict {
        padding: 8px 24px;
        border: 1px solid #B33A3A;
        background: #B33A3A;
        color: white;
        border-radius: 4px;
        cursor: pointer;
        font-family: inherit;
        font-size: 13px;
    }
    .btn-predict:hover {
        background: #7A2626;
    }
    .btn-clear:hover {
        background: #ECE8DA;
    }
    .placeholder-note {
        color: #888;
        font-style: italic;
        text-align: center;
        padding: 20px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load your existing model
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

# TARGET_SIZE = 32 (your model expects 32x32)

# --------------------------------------------------
# Your existing preprocessing function (adapted for canvas)
# --------------------------------------------------

def preprocess_canvas_image(image_array):
    """
    Preprocess canvas image using your existing pipeline.
    This matches your current preprocessing exactly.
    """
    # Canvas image is already grayscale (0-255)
    # Invert if needed (canvas draws black on white)
    
    # Apply threshold
    _, thresh = cv2.threshold(image_array, 30, 255, cv2.THRESH_BINARY_INV)
    
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
    
    # Add padding (25% of max dimension)
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
    
    # Resize to 32x32 (your model expects this)
    digit_resized = cv2.resize(digit, (32, 32), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = digit_resized.astype(np.float32) / 255.0
    
    # Reshape for CNN input
    normalized = normalized.reshape(1, 32, 32, 1)
    
    return digit_resized, normalized

# --------------------------------------------------
# HTML Drawing Canvas Component
# --------------------------------------------------

def get_canvas_html():
    """Return the HTML/JS for the drawing canvas"""
    return """
    <div class="canvas-wrapper">
        <canvas id="drawCanvas" class="drawing-canvas" width="280" height="280"></canvas>
        <div class="controls">
            <button class="btn-clear" id="clearBtn">✏️ Clear</button>
            <button class="btn-predict" id="predictBtn">🔍 Predict</button>
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
        
        // Send clear event to Streamlit
        const data = { type: 'canvas_cleared' };
        window.parent.postMessage(data, '*');
    });
    
    // Predict button
    document.getElementById('predictBtn').addEventListener('click', () => {
        if (!hasInk) {
            alert('Please draw a digit first!');
            return;
        }
        
        // Get canvas data as image
        const imageData = canvas.toDataURL('image/png');
        
        // Send to Streamlit
        const data = { 
            type: 'canvas_prediction', 
            imageData: imageData 
        };
        window.parent.postMessage(data, '*');
    });
    </script>
    """

# --------------------------------------------------
# Display probability bars
# --------------------------------------------------

def display_probability_bars(probabilities):
    """Create HTML for probability bars"""
    html = '<div style="width: 100%;">'
    for i, prob in enumerate(probabilities):
        pct = prob * 100
        # Highlight the max
        is_max = prob == max(probabilities)
        color = '#B33A3A' if is_max else '#55554F'
        html += f"""
        <div class="bar-container">
            <div class="bar-label">
                <span><strong>{i}</strong> {'⭐' if is_max else ''}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: {pct}%; background: {color};"></div>
            </div>
        </div>
        """
    html += '</div>'
    return html

# --------------------------------------------------
# Main UI
# --------------------------------------------------

st.title("🔢 Handwritten Digit Recognizer")
st.markdown("Draw a digit (0-9) in the box below or upload an image")

# Create tabs
tab1, tab2 = st.tabs(["✏️ Draw", "📤 Upload Image"])

# --------------------------------------------------
# Tab 1: Drawing Canvas
# --------------------------------------------------

with tab1:
    # Display the canvas
    st.components.v1.html(get_canvas_html(), height=380)
    
    # Initialize session state for prediction
    if 'draw_result' not in st.session_state:
        st.session_state.draw_result = None
    if 'draw_processed' not in st.session_state:
        st.session_state.draw_processed = None
    
    # Display prediction results
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        if st.session_state.draw_result is not None:
            result = st.session_state.draw_result
            confidence_class = 'confidence-high' if result['confidence'] > 80 else 'confidence-medium' if result['confidence'] > 50 else 'confidence-low'
            
            st.markdown(f"""
            <div class="prediction-box">
                <div class="digit-display">{result['digit']}</div>
                <div class="confidence-text">{result['confidence']:.1f}% sure</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show processed image
            if st.session_state.draw_processed is not None:
                st.image(st.session_state.draw_processed, width=150, clamp=True)
                st.caption("What the model sees (32×32)")
        else:
            st.markdown("""
            <div class="placeholder-note">
                ✏️ Draw a digit and click "Predict"
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.draw_result is not None:
            st.markdown("### 📊 Probability Distribution")
            st.markdown(
                display_probability_bars(st.session_state.draw_result['probabilities']),
                unsafe_allow_html=True
            )

# --------------------------------------------------
# Tab 2: Upload Image
# --------------------------------------------------

with tab2:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload an image containing a handwritten digit"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Show original with rotation controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, width=300)
        
        # Process and predict
        with st.spinner("Processing..."):
            # Convert to numpy for preprocessing
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Use the same preprocessing as canvas
            processed, model_input = preprocess_canvas_image(gray)
            
            if processed is None:
                st.error("❌ Could not detect a digit in the image.")
            else:
                # Make prediction
                if model is not None:
                    prediction = model.predict(model_input, verbose=0)
                    predicted_digit = np.argmax(prediction[0])
                    confidence = np.max(prediction[0]) * 100
                    
                    # Show results
                    col1, col2 = st.columns([1, 1.5])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="digit-display">{predicted_digit}</div>
                            <div class="confidence-text">{confidence:.1f}% sure</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(processed, width=150, clamp=True)
                        st.caption("Processed digit (32×32)")
                    
                    with col2:
                        st.markdown("### 📊 Probability Distribution")
                        st.markdown(
                            display_probability_bars(prediction[0]),
                            unsafe_allow_html=True
                        )

# --------------------------------------------------
# JavaScript to handle canvas messages
# --------------------------------------------------

# This JavaScript captures messages from the canvas and processes them
js_code = """
<script>
window.addEventListener('message', function(event) {
    // Check if the message is from the canvas
    if (event.data && event.data.type === 'canvas_prediction') {
        // Send the image data to Streamlit via the backend
        const imageData = event.data.imageData;
        
        // We'll use fetch to send the image to the server
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
    }
});
</script>
"""
st.components.v1.html(js_code, height=0)

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
    
    **Accuracy:** High accuracy on MNIST-like digits
    """)
    
    st.divider()
    
    st.header("💡 Tips")
    st.markdown("""
    1. ✍️ Draw **boldly** with good contrast
    2. 🎯 Center the digit
    3. ✏️ Use simple print style (not cursive)
    4. 🔄 Use **Clear** to start over
    5. 📤 Upload images for batch testing
    """)
    
    st.divider()
    
    if model is not None:
        st.success("✅ Model ready!")
        st.info(f"📐 Input size: 32×32 pixels")
    else:
        st.error("❌ Model not loaded")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
