import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image, ImageOps


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "handwritten_digit_cnn.keras"
    )


model = load_model()


# --------------------------------------------------
# Preprocess image
# --------------------------------------------------

def preprocess_image(image):
    """
    Convert uploaded image into the same format
    used during CNN training.
    """

    # Convert PIL image to RGB
    image = image.convert("RGB")

    # Convert PIL -> NumPy
    image = np.array(image)

    # RGB -> grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # --------------------------------------------------
    # Improve contrast
    # --------------------------------------------------

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # --------------------------------------------------
    # Threshold
    # --------------------------------------------------

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --------------------------------------------------
    # Find digit
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None

    # Find the largest contour
    contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(contour)

    # --------------------------------------------------
    # Crop digit
    # --------------------------------------------------

    digit = thresh[
        y:y + h,
        x:x + w
    ]

    # --------------------------------------------------
    # Add padding
    # --------------------------------------------------

    padding = int(
        max(w, h) * 0.25
    )

    digit = cv2.copyMakeBorder(
        digit,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # --------------------------------------------------
    # Resize while keeping digit proportions
    # --------------------------------------------------

    h2, w2 = digit.shape

    scale = 28 / max(
        h2,
        w2
    )

    new_w = max(
        1,
        int(w2 * scale)
    )

    new_h = max(
        1,
        int(h2 * scale)
    )

    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------
    # Put digit in 32x32 canvas
    # --------------------------------------------------

    canvas = np.zeros(
        (32, 32),
        dtype=np.uint8
    )

    start_x = (32 - new_w) // 2
    start_y = (32 - new_h) // 2

    canvas[
        start_y:start_y + new_h,
        start_x:start_x + new_w
    ] = digit

    # --------------------------------------------------
    # Normalize to [0, 1]
    # --------------------------------------------------

    normalized = canvas.astype(
        np.float32
    ) / 255.0

    # Add CNN dimensions
    normalized = normalized.reshape(
        1,
        32,
        32,
        1
    )

    return canvas, normalized


# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title("Handwritten Digit Recognition")

st.write(
    "Upload an image containing a handwritten digit "
    "from 0 to 9."
)


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a handwritten digit",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    # --------------------------------------------------
    # Open image
    # --------------------------------------------------

    image = Image.open(
        uploaded_file
    )

    # --------------------------------------------------
    # FIX CAMERA EXIF ORIENTATION
    #
    # This does NOT force a 90-degree rotation.
    # It only applies the orientation information
    # stored by the camera.
    # --------------------------------------------------

    image = ImageOps.exif_transpose(
        image
    )

    # Convert to RGB
    image = image.convert(
        "RGB"
    )


    # --------------------------------------------------
    # Show ORIGINAL image
    # --------------------------------------------------

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        width=400
    )


    # --------------------------------------------------
    # Preprocess
    # --------------------------------------------------

    processed_image, model_input = preprocess_image(
        image
    )


    # --------------------------------------------------
    # Check whether digit was detected
    # --------------------------------------------------

    if processed_image is None:

        st.error(
            "Could not detect a digit in the image."
        )

    else:

        # --------------------------------------------------
        # Show processed image
        # --------------------------------------------------

        st.subheader(
            "Processed Image"
        )

        st.image(
            processed_image,
            width=300,
            clamp=True
        )


        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = model.predict(
            model_input,
            verbose=0
        )

        predicted_digit = np.argmax(
            prediction[0]
        )

        confidence = (
            np.max(prediction[0]) * 100
        )


        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        st.subheader(
            "Prediction"
        )

        st.success(
            f"Predicted Digit: {predicted_digit}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )


        # --------------------------------------------------
        # Show probabilities
        # --------------------------------------------------

        st.subheader(
            "Prediction Probabilities"
        )

        probabilities = prediction[0]

        for digit, probability in enumerate(
            probabilities
        ):

            st.write(
                f"Digit {digit}: "
                f"{probability * 100:.2f}%"
            )