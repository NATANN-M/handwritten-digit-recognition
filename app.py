import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image



@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "handwritten_digit_cnn.keras"
    )

    return model


model = load_model()


# ==========================================
# Preprocess uploaded image
# ==========================================

def preprocess_image(image):

    # PIL image -> NumPy
    image = np.array(image)

    # RGB -> grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

  

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)


    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        10
    )


    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    clean = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )



    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_height, image_width = clean.shape

    image_area = (
        image_height * image_width
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(
            contour
        )

        box_area = w * h

        # Ignore very small objects
        if area < 100:
            continue

        # Ignore huge background regions
        if box_area > image_area * 0.50:
            continue

        # Ignore very small boxes
        if w < 10 or h < 10:
            continue

        # Ignore extremely thin objects
        aspect_ratio = w / h

        if aspect_ratio > 10 or aspect_ratio < 0.1:
            continue

        candidates.append(
            (area, x, y, w, h)
        )

    # No digit detected
    if len(candidates) == 0:

        return None, None

    # Largest meaningful contour
    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    area, x, y, w, h = candidates[0]

 
    padding = int(
        max(w, h) * 0.15
    )

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        image_width,
        x + w + padding
    )

    y2 = min(
        image_height,
        y + h + padding
    )

    cropped = enhanced[
        y1:y2,
        x1:x2
    ]


    height, width = cropped.shape

    size = max(
        height,
        width
    )

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    y_offset = (
        size - height
    ) // 2

    x_offset = (
        size - width
    ) // 2

    square[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = cropped

  

    resized = cv2.resize(
        square,
        (32, 32),
        interpolation=cv2.INTER_AREA
    )



    normalized = (
        resized.astype(np.float32)
        / 255.0
    )

  

    model_input = normalized.reshape(
        1,
        32,
        32,
        1
    )

    return model_input, resized

# ==========================================
# Streamlit interface
# ==========================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢"
)

st.title(
    "Handwritten Digit Recognition"
)

st.write(
    "Upload an image of a handwritten digit "
    "from 0 to 9."
)


# ==========================================
# Upload image
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a handwritten digit",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==========================================
# Process uploaded image
# ==========================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        width=300
    )

    if st.button(
        "Recognize Digit"
    ):

        processed, resized = preprocess_image(
            image
        )

        if processed is None:

            st.error(
                "Could not detect a digit "
                "in the uploaded image."
            )

        else:

            # ==================================
            # Show processed image
            # ==================================

            st.subheader(
                "Processed Image"
            )

            st.image(
                resized,
                width=200
            )

            # ==================================
            # Predict
            # ==================================

            predictions = model.predict(
                processed,
                verbose=0
            )

            predicted_digit = int(
                np.argmax(
                    predictions[0]
                )
            )

            confidence = float(
                np.max(
                    predictions[0]
                )
            )

            # ==================================
            # Result
            # ==================================

            st.success(
                f"Predicted Digit: {predicted_digit}"
            )

            st.info(
                f"Confidence: "
                f"{confidence * 100:.2f}%"
            )

            # ==================================
            # All probabilities
            # ==================================

            st.subheader(
                "Prediction Probabilities"
            )

            for digit, probability in enumerate(
                predictions[0]
            ):

                st.write(
                    f"Digit {digit}: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )